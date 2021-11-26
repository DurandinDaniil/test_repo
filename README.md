## examples

### get frame grid

```python
import cv2
from lattice.tracker import mask2points

p = 'data/c167_2803_15_18/c167_2803_15_180001.png'
frame = cv2.imread(filename=p)

grid = mask2points(frame)
indexes, points = grid.indexes, grid.points
 

```


### update frame grid

Look at `example.py` for details.