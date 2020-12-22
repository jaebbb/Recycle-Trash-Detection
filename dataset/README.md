## Prepare datasets

It is recommended to dataset root to `$Recycle-Trash-Detection/dataset`.  
If your folder structure is different, you may need to change the corresponding paths in files.

```
Recycle-Trash-Detection
├── dataset
│   ├── train
│   │   ├── imgs
│   │   │   ├── image1.jpg
│   │   │   ├── .....
│   │   ├── json
│   │   │   ├── image1.json
│   │   │   ├── .....
│   ├── val
│   │   ├── imgs
│   │   │   ├── image1.jpg
│   │   │   ├── .....
│   │   ├── json
│   │   │   ├── image1.json
│   │   │   ├── .....
│   ├── test
│   │   ├── imgs
│   │   │   ├── image1.jpg
│   │   │   ├── .....
```  

### Json Format  
```
{
    "filename": "img1.jpg",
    "object": [
        {
            "label": "label_name",
            "points": [ 
                [,], [,], [,], [,]
            ],
            "group_id": null,
            "shape_type": "polygon",
            "inner": null,
            "flags": {}
        }
    ]
}
```