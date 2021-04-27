from pydantic import BaseModel
from typing import List


class Item(BaseModel):
    url: str


class Image(BaseModel):
    uuid: str
    url: str
    height: int
    width: int


class InspectionViewImages(BaseModel):
    images: List[Image]


class ImageAnnotations(BaseModel):
    uuid: str
    annotations: List[List]


class InspectionViewAnnotations(BaseModel):
    dataset_uuid: str
    image_annotations_list: List[ImageAnnotations]
