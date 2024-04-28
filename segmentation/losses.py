from monai.losses.hausdorff_loss import HausdorffDTLoss

import warnings
from typing import Callable

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.transforms.utils import distance_transform_edt
from monai.utils import LossReduction


class HausdorffDTLoss(_Loss):
    """
    Compute channel-wise binary Hausdorff loss based on distance transform. It can support both multi-classes and
    multi-labels tasks. The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
    (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper: Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with
    Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
    """

    def __init__(
        self,
        alpha: float = 2.0,
        include_background: bool = False,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super(HausdorffDTLoss, self).__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.alpha = alpha
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch


    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        """Generate distance transform.

        Args:
            img (np.ndarray): input mask as NCHWD or NCHW.

        Returns:
            np.ndarray: Distance field.
        """
        field = torch.zeros_like(img)

        for batch_idx in range(len(img)):
            fg_mask = img[batch_idx] > 0.5

            # For cases where the mask is entirely background or entirely foreground
            # the distance transform is not well defined for all 1s,
            # which always would happen on either foreground or background, so skip
            if fg_mask.any() and not fg_mask.all():
                fg_dist: torch.Tensor = distance_transform_edt(fg_mask)  # type: ignore
                bg_mask = ~fg_mask
                bg_dist: torch.Tensor = distance_transform_edt(bg_mask)  # type: ignore

                field[batch_idx] = fg_dist + bg_dist

        return field


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D], where N is the number of classes.
            target: the shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Raises:
            ValueError: If the input is not 2D (NCHW) or 3D (NCHWD).
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError("Only 2D (NCHW) and 3D (NCHWD) supported")

        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # If skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        device = input.device
        all_f = []
        for i in range(input.shape[1]):
            ch_input = input[:, [i]]
            ch_target = target[:, [i]]
            pred_dt = self.distance_field(ch_input.detach()).float()
            target_dt = self.distance_field(ch_target.detach()).float()

            pred_error = (ch_input - ch_target) ** 2
            distance = pred_dt**self.alpha + target_dt**self.alpha

            running_f = pred_error * distance.to(device)
            reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
            if self.batch:
                # reducing spatial dimensions and batch
                reduce_axis = [0] + reduce_axis
            all_f.append(running_f.mean(dim=reduce_axis, keepdim=True))
        f = torch.cat(all_f, dim=1)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least make sure a none reduction maintains a
            # broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(ch_input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f



class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.weight = weight
        self.register_buffer("class_weight", torch.ones(1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.weight is not None and target.shape[1] != 1:
            # make sure the lengths of weights are equal to the number of classes
            num_of_classes = target.shape[1]
            if isinstance(self.weight, (float, int)):
                self.class_weight = torch.as_tensor([self.weight] * num_of_classes)
            else:
                self.class_weight = torch.as_tensor(self.weight)
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    
    
class CombinedHausdorffDiceLoss(_Loss):
    """
    Compute a combination of Dice and Hausdorff distance losses.
    """

    def __init__(
        self,
        hausdorff_weight: float = 0.5,
        dice_weight: float = 0.5,
        hausdorff_kwargs: dict = {},
        dice_kwargs: dict = {},
    ):
        """
        Args:
            hausdorff_weight: Weight for the Hausdorff distance loss.
            dice_weight: Weight for the Dice loss.
            hausdorff_kwargs: Additional arguments for the HausdorffDTLoss class.
            dice_kwargs: Additional arguments for the DiceLoss class.
        """
        super().__init__()
        self.hausdorff_loss = HausdorffDTLoss(**hausdorff_kwargs)
        self.dice_loss = DiceLoss(**dice_kwargs)
        self.hausdorff_weight = hausdorff_weight
        self.dice_weight = dice_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the predictions from the network (logits or probabilities).
            target: the ground truth labels.

        Returns:
            Combined loss value.
        """
        hausdorff_loss = self.hausdorff_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        combined_loss = self.hausdorff_weight * hausdorff_loss + self.dice_weight * dice_loss
        return combined_loss


#Bounding the hausdorff to 0-1

import torch

def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hausdorff_loss = self.hausdorff_loss(input, target)
    
    # Initialize a tensor to store the maximum diameters
    max_diameters = torch.zeros(target.shape[0], device=target.device)
    
    # Compute the diameter for each object in the batch
    for i in range(target.shape[0]):
        # Compute the distance transform for each target mask
        # Note: You might need to adapt this based on how your target masks are structured
        dist_transform = torch.nn.functional.distance_transform_edt(target[i] > 0.5)
        
        # The diameter is twice the maximum value in the distance transform
        max_diameters[i] = torch.max(dist_transform) * 2

    # Normalize the Hausdorff loss by the diameters
    # Avoid division by zero by adding a small epsilon
    normalized_hausdorff_loss = hausdorff_loss / (max_diameters + 1e-6)
    
    # Clamp values to [0, 1] to handle any potential numerical instability
    normalized_hausdorff_loss = torch.clamp(normalized_hausdorff_loss, min=0, max=1)

    dice_loss = self.dice_loss(input, target)
    combined_loss = self.hausdorff_weight * normalized_hausdorff_loss + self.dice_weight * dice_loss
    return combined_loss

