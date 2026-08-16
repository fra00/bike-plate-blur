# ─── Global constants ────────────────────────────────────────────────────────
import os
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── YOLO vehicle class IDs (COCO dataset) ───────────────────────────────────
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

VEHICLE_FILTER_MAP = {
    "all":       {2, 3, 5, 7},
    "motorbike": {3},
    "car":       {2},
    "bus":       {5},
    "truck":     {7},
}

PLATE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "license-plate-finetune-v1m.pt")
DETECT_WIDTH = 1280  # both models run at this width; coords scaled back to full-res

# TensorRT engine batch size (exported with batch=4). The engine accepts
# exactly this many images per call, so tiles/crops are padded to multiples.
TRT_BATCH = 4

_CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1":  "av1_cuvid",
    "vp9":  "vp9_cuvid",
}

_DBG_VEHICLE_COLOR = (255, 100,   0)   # blue
_DBG_GHOST_COLOR      = (180,  80,  80)   # dim teal — tracked vehicle, detector missed
_DBG_SUPPRESSED_COLOR = (120, 120, 120)   # grey    — duplicate plate, suppressed
_DBG_PLATE_COLOR      = (  0, 220,   0)   # green  — raw model detection
_DBG_PREDICT_COLOR = (  0, 220, 220)   # yellow — tracker-predicted (gap fill)
_DBG_BLUR_COLOR    = (  0,   0, 220)   # red    — padded blur region
_DBG_OWN_COLOR     = (  0, 140, 255)   # orange — own plate fixed region
_DBG_FONT          = cv2.FONT_HERSHEY_SIMPLEX

# Brand palette (BGR for OpenCV).  Kept in sync with the dsdt.x assets.
_DD_VOID_BG   = (15,  15,  18)
_DD_HUD_BG    = (18,  18,  24)
_DD_HUD_LINE  = (40,  40,  50)
_DD_WHITE     = (240, 240, 240)
_DD_DIM       = (140, 140, 150)
_DD_RED       = (26,  0,   226)    # Signal Red
_DD_BLUE      = (255, 102, 0)      # Data Blue
_DD_GREEN     = (0,   230, 0)      # detection box
_DD_YELLOW    = (0,   220, 220)    # predicted (gap-fill)
_DD_ORANGE    = (0,   140, 255)    # own-plate
_DD_TRAIL     = (255, 200, 80)     # cyan trajectory tail
_DD_GHOST     = (90,  90,  100)    # rejected / dropped

# Source-to-colour map for plate boxes
_DD_SOURCE_COLOR = {
    "sahi": _DD_GREEN,
    "crop": _DD_GREEN,
    "pred": _DD_YELLOW,
    "own":  _DD_ORANGE,
    "fallback": (255, 0, 255),   # magenta — privacy fallback strip
    "anchor": (255, 255, 0),     # cyan — anchor-only moto zone
    "patch": (0, 200, 255),      # amber — guaranteed-coverage patch over a
                                 # detection the smoothed zone does not cover
}

_DD_FONT_DIR = os.path.join(os.path.expanduser("~"),
                             ".local/share/fonts/dsdtx")
_DD_F_DISP  = os.path.join(_DD_FONT_DIR, "OrbitronVar.ttf")
_DD_F_BODY  = os.path.join(_DD_FONT_DIR, "Rajdhani-SemiBold.ttf")
_DD_F_MONO  = os.path.join(_DD_FONT_DIR, "JetBrainsMono-Regular.ttf")
_DD_F_MONOB = os.path.join(_DD_FONT_DIR, "JetBrainsMono-Bold.ttf")
_DD_HUD_W   = 320
