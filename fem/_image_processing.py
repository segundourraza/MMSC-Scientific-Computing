"""
image_to_grid_and_mesh.py

Dependencies:
    pip install pillow numpy scipy matplotlib

Functions:
    parse_image_to_grid(image_path, mask_path=None, damage_color=None, white_thresh=0.9, black_thresh=0.1)
    grid_to_mesh(values, spacing=(1.0,1.0), delaunay=True)
    Example usage at bottom.
"""

from PIL import Image
import numpy as np
from scipy.spatial import Delaunay
from scipy import ndimage
import matplotlib.pyplot as plt


def _normalize_img(img):
    """Return image as float32 array in [0,1] with shape (H,W,3) or (H,W)."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        # try to scale if out of [0,1]
        if arr.max() > 1.0:
            arr = arr / arr.max()
    return arr


def parse_image_to_grid(
    image_path,
    mask_path=None,
    damage_color=None,
    white_thresh=0.9,
    black_thresh=0.1,
    damage_tol=30,
    clean_damage=True,
    remove_small_holes_size=8
):
    """
    Parse image to a values grid with -1 (black), +1 (white), and 0 (damage).

    Parameters
    ----------
    image_path : str
        Path to PNG/JPEG image.
    mask_path : str or None
        Optional path to a separate mask image (nonzero = damage).
    damage_color : tuple (R,G,B) or None
        If provided, pixels near this color are treated as damage.
        Example: (255,0,0) to use red-painted damage regions.
    white_thresh : float
        Threshold in [0,1] above which a pixel is considered white.
        Applied to grayscale intensity.
    black_thresh : float
        Threshold in [0,1] below which a pixel is considered black.
    damage_tol : float
        Color distance tolerance (0-255 scale) when matching damage_color.
    clean_damage : bool
        If True, perform small morphological cleaning on mask.
    remove_small_holes_size : int
        Size threshold (in pixels) to remove tiny holes in damage mask.

    Returns
    -------
    values : 2D numpy array (H,W), dtype=float32
        Grid of -1, 0, 1.
    meta : dict
        Metadata: {'height', 'width', 'spacing'}
    """
    img = Image.open(image_path)
    img = img.convert("RGBA")  # keep alpha if present
    arr = _normalize_img(img)  # shape (H,W,4)
    H, W = arr.shape[:2]

    # 1) detect damage mask from mask_path if given
    damage_mask = np.zeros((H, W), dtype=bool)
    if mask_path:
        m = Image.open(mask_path).convert("L")
        m_arr = _normalize_img(m)
        damage_mask = (m_arr > 0.5)
    else:
        # 2) try alpha channel (transparency)
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            # consider fully transparent or nearly transparent as damage
            damage_mask = damage_mask | (alpha < 0.99)

        # 3) match damage_color if given
        if damage_color is not None:
            # convert damage_color (0-255) to normalized
            dc = np.array(damage_color, dtype=float) / 255.0
            rgb = arr[..., :3]
            # Euclidean distance in RGB
            dist = np.linalg.norm(rgb - dc, axis=2)
            # damage color tolerance converted to normalized scale:
            damage_mask = damage_mask | (dist < (damage_tol / 255.0))

    # optional small cleaning of damage mask (remove tiny islands)
    if clean_damage and damage_mask.any():
        # remove small objects and fill small holes
        damage_mask = ndimage.binary_opening(damage_mask, structure=np.ones((3,3)))
        damage_mask = ndimage.binary_closing(damage_mask, structure=np.ones((3,3)))
        # remove small connected components
        labeled, n = ndimage.label(damage_mask)
        sizes = ndimage.sum(damage_mask, labeled, range(1, n+1))
        mask_sizes = np.zeros(n+1, dtype=int)
        mask_sizes[1:] = sizes
        # keep components larger than threshold
        keep = np.isin(labeled, np.where(mask_sizes > remove_small_holes_size)[0])
        damage_mask = keep

    # 4) compute grayscale intensity for white/black detection
    rgb = arr[..., :3]
    # use standard luminance formula if RGB
    luminance = np.dot(rgb, [0.299, 0.587, 0.114])  # shape (H,W)

    # build values: default = -1 (black)
    values = np.full((H, W), fill_value=np.nan, dtype=np.float32)

    # damage → 0 (highest priority)
    values[damage_mask] = 0.0

    # white pixels → +1
    white_pixels = (luminance >= white_thresh) & (~damage_mask)
    values[white_pixels] = +1.0

    # black pixels → -1
    black_pixels = (luminance <= black_thresh) & (~damage_mask)
    values[black_pixels] = -1.0

    # remaining pixels (mid-grey / unknown): assign by nearest neighbor or threshold to -1/+1
    unknown = np.isnan(values)
    if unknown.any():
        # simple rule: threshold at 0.5 mid-grey
        values[unknown & (luminance > 0.5)] = +1.0
        values[unknown & (luminance <= 0.5)] = -1.0

    meta = {"height": H, "width": W, "spacing": (1.0, 1.0)}
    return values, meta


def grid_to_mesh(values, spacing=(1.0, 1.0), delaunay=True):
    """
    Convert regular pixel grid to node list and optionally Delaunay triangles.

    Returns:
        nodes : (N,2) float array with (x,y) coordinates (pixel centers)
        vals  : (N,) array of values mapped from the input grid
        tri   : scipy.spatial.Delaunay object or None
    """
    H, W = values.shape
    sx, sy = spacing
    # pixel centers: x across columns, y down rows
    xs = (np.arange(W) + 0.5) * sx
    ys = (np.arange(H) + 0.5) * sy
    xv, yv = np.meshgrid(xs, ys)
    nodes = np.column_stack([xv.ravel(), yv.ravel()])
    vals = values.ravel().astype(np.float32)

    tri = None
    if delaunay:
        # Delaunay on node coordinates (may be heavy for large images)
        tri = Delaunay(nodes)
    return nodes, vals, tri


def show_grid(values, title="parsed values"):
    plt.figure(figsize=(6,6))
    # display values as image with -1 -> black, 0 -> gray, +1 -> white
    cmap = plt.get_cmap("gray")
    plt.imshow(values, origin="upper", vmin=-1, vmax=1, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="value")
    plt.axis("off")
    plt.show()


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":

    file  = "images\cross_circle.png"
    # Example 1: detect damage by red color painted on image
    values, meta = parse_image_to_grid(
        file,
        mask_path=None,
        damage_color=(255, 0, 0),
        white_thresh=0.92,
        black_thresh=0.08,
        damage_tol=40
    )
    print("shape:", values.shape)
    show_grid(values, title="Parsed +1/0/-1 grid (red damage)")

    # # Example 2: image with separate mask
    # # mask.png should be same size with damage areas white.
    # values2, meta2 = parse_image_to_grid("photo.png", mask_path="mask.png")
    # show_grid(values2, "Using external mask")

    # Example 3: get mesh
    nodes, vals, tri = grid_to_mesh(values, spacing=(1.0, 1.0), delaunay=False)
    print("nodes:", nodes.shape, "vals:", vals.shape)
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    tri = Triangulation(nodes, tri.simplices)
    plt.triplot(tri)
    if tri is not None:
        print("triangles:", tri.simplices.shape)