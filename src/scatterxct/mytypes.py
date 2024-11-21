from typing import Any, TypeVar, Union

import numpy as np
import numpy.typing as npt
from nptyping import Complex128, Float64, NDArray, Shape

Array = npt.NDArray
RealArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
AnyArray = npt.NDArray[Any]

A = TypeVar("A")
B = TypeVar("B")
M = TypeVar("M")
N = TypeVar("N")

# Shape types
ShapeOperator = Shape["A, A"]
ShapeVector = Shape["A"]
ShapeVectorOperator = Shape["A, A, B"]

# Custom numpy object types

RealOperator = NDArray[Shape["A, A"], Float64]
ComplexOperator = NDArray[Shape["A, A"], Complex128]
GenericOperator = Union[RealOperator, ComplexOperator]

RealVector = NDArray[Shape["A"], Float64]
ComplexVector = NDArray[Shape["A"], Complex128]
GenericVector = Union[RealVector, ComplexVector]

RealVectorOperator = NDArray[Shape["A, A, B"], Float64]
ComplexVectorOperator = NDArray[Shape["A, A, B"], Complex128]
GenericVectorOperator = Union[RealVectorOperator, ComplexVectorOperator]

RealDiagonalVectorOperator = NDArray[Shape["A, B"], Float64]
ComplexDiagonalVectorOperator = NDArray[Shape["A, B"], Complex128]
GenericDiagonalVectorOperator = Union[RealDiagonalVectorOperator, ComplexDiagonalVectorOperator]
