import cv2
import numpy as np
import time
import os
import colorsys
from datetime import datetime
from collections import Counter, deque

class YarnColorDatabase:
    """Enhanced database with optimized color definitions for yarn detection"""
    
    # Basic 8 colors - these are the highest priority for detection
    BASIC_COLORS = {
        'Brown': {
            'code': 'YC-010', 
            'name': 'Chocolate Brown',
            'lab': (35.0, 25.0, 35.0),
            'rgb': (139, 69, 19),
            'hsv': (20, 86, 55),
            'ranges': {
                'hsv_lower': (10, 40, 30),
                'hsv_upper': (30, 100, 80)
            }
        },
        'Green': {
            'code': 'YC-005', 
            'name': 'Forest Green',
            'lab': (48.0, -55.0, 40.0),
            'rgb': (34, 139, 34),
            'hsv': (120, 76, 55),
            'ranges': {
                'hsv_lower': (100, 50, 30),
                'hsv_upper': (140, 100, 70)
            }
        },
        'Orange': {
            'code': 'YC-007', 
            'name': 'Sunset Orange',
            'lab': (70.0, 35.0, 75.0),
            'rgb': (255, 140, 0),
            'hsv': (33, 100, 100),
            'ranges': {
                'hsv_lower': (15, 70, 70),  # Modified for better detection
                'hsv_upper': (35, 100, 100)
            },
            # Direct RGB checks for orange
            'rgb_checks': {
                'min_r': 180,
                'min_g': 80,
                'max_g': 180,
                'max_b': 100,
                'r_g_ratio': 1.3  # R should be at least 1.3x G
            }
        },
        'Pink': {
            'code': 'YC-009', 
            'name': 'Rose Pink',
            'lab': (65.0, 52.0, -5.0),
            'rgb': (255, 105, 180),
            'hsv': (330, 59, 100),
            'ranges': {
                'hsv_lower': (325, 35, 70),
                'hsv_upper': (340, 70, 100)
            }
        },
        'Red': {
            'code': 'YC-003', 
            'name': 'Cherry Red',
            'lab': (45.0, 68.0, 38.0),
            'rgb': (220, 20, 60),
            'hsv': (348, 91, 86),
            'ranges': {
                'hsv_lower': (340, 70, 60),
                'hsv_upper': (360, 100, 100)
            }
        },
        'Cream': {
            'code': 'YC-012', 
            'name': 'Vanilla Cream',
            'lab': (96.0, -3.0, 15.0),
            'rgb': (255, 248, 220),
            'hsv': (48, 14, 100),
            'ranges': {
                'hsv_lower': (30, 5, 85),
                'hsv_upper': (60, 25, 100)
            },
            'rgb_thresholds': {  # Special RGB thresholds for cream detection
                'min_r': 220,
                'min_g': 220,
                'min_b': 180,
                'max_diff': 40
            }
        },
        'Yellow': {
            'code': 'YC-006', 
            'name': 'Sunshine Yellow',
            'lab': (90.0, -5.0, 85.0),
            'rgb': (255, 215, 0),
            'hsv': (51, 100, 100),
            'ranges': {
                'hsv_lower': (38, 70, 70),  # Modified for better detection
                'hsv_upper': (65, 100, 100)
            },
            # Direct RGB checks for yellow
            'rgb_checks': {
                'min_r': 180,
                'min_g': 150,
                'max_b': 120,
                'r_g_similarity': 50,  # R and G should be within 50 of each other
                'r_b_diff': 80  # R should be at least 80 more than B
            }
        },
        'Blue': {
            'code': 'YC-004', 
            'name': 'Royal Blue',
            'lab': (40.0, 22.0, -45.0),
            'rgb': (0, 102, 204),
            'hsv': (210, 100, 80),
            'ranges': {
                'hsv_lower': (200, 60, 50),
                'hsv_upper': (220, 100, 100)
            }
        }
    }
    
    # All colors (basic + advanced)
    COLORS = {
        # ======== BROWNS ========
        'Brown': BASIC_COLORS['Brown'],
        'DarkBrown': {
            'code': 'YC-103', 
            'name': 'Dark Brown',
            'lab': (25.0, 20.0, 30.0),
            'rgb': (101, 67, 33),
            'hsv': (30, 67, 40),
            'ranges': {
                'hsv_lower': (15, 50, 20),
                'hsv_upper': (35, 90, 50)
            }
        },
        'LightBrown': {
            'code': 'YC-104', 
            'name': 'Light Brown',
            'lab': (55.0, 15.0, 40.0),
            'rgb': (150, 111, 51),
            'hsv': (33, 66, 59),
            'ranges': {
                'hsv_lower': (20, 40, 40),
                'hsv_upper': (40, 80, 75)
            }
        },
        'Coffee': {
            'code': 'YC-105', 
            'name': 'Coffee Brown',
            'lab': (30.0, 22.0, 32.0),
            'rgb': (111, 78, 55),
            'hsv': (20, 50, 44),
            'ranges': {
                'hsv_lower': (10, 30, 30),
                'hsv_upper': (30, 60, 60)
            }
        },
        'Sienna': {
            'code': 'YC-212', 
            'name': 'Sienna Brown',
            'lab': (32.0, 30.0, 40.0),
            'rgb': (160, 82, 45),
            'hsv': (19, 72, 63),
            'ranges': {
                'hsv_lower': (10, 50, 40),
                'hsv_upper': (25, 90, 75)
            }
        },
        'Tan': {
            'code': 'YC-213', 
            'name': 'Tan',
            'lab': (69.0, 12.0, 36.0),
            'rgb': (210, 180, 140),
            'hsv': (34, 33, 82),
            'ranges': {
                'hsv_lower': (25, 20, 65),
                'hsv_upper': (40, 45, 90)
            }
        },
        
        # ======== REDS ========
        'Red': BASIC_COLORS['Red'],
        'BrightRed': {
            'code': 'YC-201', 
            'name': 'Bright Red',
            'lab': (50.0, 75.0, 45.0),
            'rgb': (255, 0, 0),
            'hsv': (0, 100, 100),
            'ranges': {
                'hsv_lower': (355, 80, 80),
                'hsv_upper': (10, 100, 100)
            }
        },
        'DarkRed': {
            'code': 'YC-202', 
            'name': 'Dark Red',
            'lab': (30.0, 60.0, 30.0),
            'rgb': (139, 0, 0),
            'hsv': (0, 100, 55),
            'ranges': {
                'hsv_lower': (350, 75, 30),
                'hsv_upper': (10, 100, 60)
            }
        },
        'Crimson': {
            'code': 'YC-024', 
            'name': 'Deep Crimson',
            'lab': (44.0, 72.0, 45.0),
            'rgb': (220, 20, 60),
            'hsv': (348, 91, 86),
            'ranges': {
                'hsv_lower': (340, 70, 60),
                'hsv_upper': (355, 100, 100)
            }
        },
        'Maroon': {
            'code': 'YC-014', 
            'name': 'Rich Maroon',
            'lab': (25.0, 52.0, 28.0),
            'rgb': (128, 0, 0),
            'hsv': (0, 100, 50),
            'ranges': {
                'hsv_lower': (350, 80, 25),
                'hsv_upper': (10, 100, 60)
            }
        },
        'Coral': {
            'code': 'YC-214', 
            'name': 'Coral Red',
            'lab': (65.0, 55.0, 30.0),
            'rgb': (255, 127, 80),
            'hsv': (16, 69, 100),
            'ranges': {
                'hsv_lower': (10, 50, 80),
                'hsv_upper': (20, 80, 100)
            }
        },
        
        # ======== GREENS ========
        'Green': BASIC_COLORS['Green'],
        'Lime': {
            'code': 'YC-016', 
            'name': 'Bright Lime',
            'lab': (72.0, -65.0, 70.0),
            'rgb': (50, 205, 50),
            'hsv': (120, 76, 80),
            'ranges': {
                'hsv_lower': (100, 40, 60),
                'hsv_upper': (140, 100, 100)
            }
        },
        'DarkGreen': {
            'code': 'YC-108', 
            'name': 'Dark Green',
            'lab': (35.0, -50.0, 35.0),
            'rgb': (0, 100, 0),
            'hsv': (120, 100, 39),
            'ranges': {
                'hsv_lower': (100, 70, 20),
                'hsv_upper': (140, 100, 50)
            }
        },
        'Olive': {
            'code': 'YC-023', 
            'name': 'Olive Green',
            'lab': (50.0, -25.0, 55.0),
            'rgb': (128, 128, 0),
            'hsv': (60, 100, 50),
            'ranges': {
                'hsv_lower': (50, 60, 30),
                'hsv_upper': (70, 100, 70)
            }
        },
        'Mint': {
            'code': 'YC-215', 
            'name': 'Mint Green',
            'lab': (85.0, -40.0, 20.0),
            'rgb': (152, 255, 152),
            'hsv': (120, 40, 100),
            'ranges': {
                'hsv_lower': (100, 20, 70),
                'hsv_upper': (140, 50, 100)
            }
        },
        'Emerald': {
            'code': 'YC-216', 
            'name': 'Emerald Green',
            'lab': (50.0, -70.0, 20.0),
            'rgb': (0, 128, 0),
            'hsv': (120, 100, 50),
            'ranges': {
                'hsv_lower': (110, 80, 30),
                'hsv_upper': (130, 100, 70)
            }
        },
        
        # ======== PINKS ========
        'Pink': BASIC_COLORS['Pink'],
        'LightPink': {
            'code': 'YC-112', 
            'name': 'Light Pink',
            'lab': (80.0, 30.0, -5.0),
            'rgb': (255, 182, 193),
            'hsv': (351, 29, 100),
            'ranges': {
                'hsv_lower': (340, 15, 80),
                'hsv_upper': (360, 40, 100)
            }
        },
        'HotPink': {
            'code': 'YC-113', 
            'name': 'Hot Pink',
            'lab': (60.0, 70.0, -10.0),
            'rgb': (255, 105, 180),
            'hsv': (330, 59, 100),
            'ranges': {
                'hsv_lower': (320, 40, 80),
                'hsv_upper': (340, 75, 100)
            }
        },
        'DeepPink': {
            'code': 'YC-114', 
            'name': 'Deep Pink',
            'lab': (55.0, 80.0, -5.0),
            'rgb': (255, 20, 147),
            'hsv': (328, 92, 100),
            'ranges': {
                'hsv_lower': (320, 70, 70),
                'hsv_upper': (340, 100, 100)
            }
        },
        'Fuchsia': {
            'code': 'YC-217', 
            'name': 'Fuchsia Pink',
            'lab': (60.0, 85.0, -20.0),
            'rgb': (255, 0, 255),
            'hsv': (300, 100, 100),
            'ranges': {
                'hsv_lower': (290, 70, 80),
                'hsv_upper': (310, 100, 100)
            }
        },
        'RosePink': {
            'code': 'YC-218', 
            'name': 'Rose Pink',
            'lab': (75.0, 40.0, -5.0),
            'rgb': (255, 150, 180),
            'hsv': (340, 41, 100),
            'ranges': {
                'hsv_lower': (330, 30, 80),
                'hsv_upper': (350, 60, 100)
            }
        },
        
        # ======== CREAMS/BEIGE ========
        'Cream': BASIC_COLORS['Cream'],
        'Beige': {
            'code': 'YC-022', 
            'name': 'Natural Beige',
            'lab': (94.0, -1.0, 20.0),
            'rgb': (245, 245, 220),
            'hsv': (60, 10, 96),
            'ranges': {
                'hsv_lower': (35, 5, 80),
                'hsv_upper': (70, 25, 100)
            },
            'rgb_thresholds': {
                'min_r': 220,
                'min_g': 220,
                'min_b': 180,
                'max_diff': 50
            }
        },
        'LightCream': {
            'code': 'YC-117', 
            'name': 'Light Cream',
            'lab': (97.0, -2.0, 10.0),
            'rgb': (255, 250, 240),
            'hsv': (40, 6, 100),
            'ranges': {
                'hsv_lower': (25, 0, 90),
                'hsv_upper': (60, 20, 100)
            },
            'rgb_thresholds': {
                'min_r': 235,
                'min_g': 235,
                'min_b': 220,
                'max_diff': 30
            }
        },
        'Ivory': {
            'code': 'YC-219', 
            'name': 'Ivory',
            'lab': (97.0, -1.0, 8.0),
            'rgb': (255, 255, 240),
            'hsv': (60, 6, 100),
            'ranges': {
                'hsv_lower': (40, 0, 90),
                'hsv_upper': (70, 15, 100)
            },
            'rgb_thresholds': {
                'min_r': 240,
                'min_g': 240,
                'min_b': 230,
                'max_diff': 20
            }
        },
        'Ecru': {
            'code': 'YC-220', 
            'name': 'Ecru',
            'lab': (88.0, 2.0, 25.0),
            'rgb': (240, 234, 190),
            'hsv': (54, 21, 94),
            'ranges': {
                'hsv_lower': (45, 15, 85),
                'hsv_upper': (65, 30, 100)
            }
        },
        
        # ======== BLUES ========
        'Blue': BASIC_COLORS['Blue'],
        'Navy': {
            'code': 'YC-013', 
            'name': 'Deep Navy',
            'lab': (20.0, 32.0, -62.0),
            'rgb': (0, 0, 128),
            'hsv': (240, 100, 50),
            'ranges': {
                'hsv_lower': (230, 70, 20),
                'hsv_upper': (250, 100, 60)
            }
        },
        'LightBlue': {
            'code': 'YC-121', 
            'name': 'Light Blue',
            'lab': (75.0, -10.0, -30.0),
            'rgb': (173, 216, 230),
            'hsv': (195, 25, 90),
            'ranges': {
                'hsv_lower': (185, 10, 75),
                'hsv_upper': (205, 40, 100)
            }
        },
        'SkyBlue': {
            'code': 'YC-221', 
            'name': 'Sky Blue',
            'lab': (80.0, -15.0, -25.0),
            'rgb': (135, 206, 235),
            'hsv': (197, 43, 92),
            'ranges': {
                'hsv_lower': (190, 30, 80),
                'hsv_upper': (210, 60, 100)
            }
        },
        'Turquoise': {
            'code': 'YC-222', 
            'name': 'Turquoise Blue',
            'lab': (70.0, -30.0, -15.0),
            'rgb': (64, 224, 208),
            'hsv': (174, 71, 88),
            'ranges': {
                'hsv_lower': (165, 50, 70),
                'hsv_upper': (180, 90, 100)
            }
        },
        'Denim': {
            'code': 'YC-223', 
            'name': 'Denim Blue',
            'lab': (40.0, -10.0, -40.0),
            'rgb': (21, 96, 189),
            'hsv': (218, 89, 74),
            'ranges': {
                'hsv_lower': (210, 70, 50),
                'hsv_upper': (230, 100, 90)
            }
        },
        
        # ======== YELLOWS/ORANGES ========
        'Yellow': BASIC_COLORS['Yellow'],
        'Orange': BASIC_COLORS['Orange'],
        'Gold': {
            'code': 'YC-020', 
            'name': 'Golden Yellow',
            'lab': (88.0, 8.0, 82.0),
            'rgb': (255, 215, 0),
            'hsv': (51, 100, 100),
            'ranges': {
                'hsv_lower': (40, 75, 75),
                'hsv_upper': (55, 100, 100)
            }
        },
        'Amber': {
            'code': 'YC-224', 
            'name': 'Amber',
            'lab': (75.0, 20.0, 80.0),
            'rgb': (255, 191, 0),
            'hsv': (45, 100, 100),
            'ranges': {
                'hsv_lower': (35, 85, 80),
                'hsv_upper': (50, 100, 100)
            }
        },
        'Peach': {
            'code': 'YC-225', 
            'name': 'Peach',
            'lab': (80.0, 20.0, 30.0),
            'rgb': (255, 218, 185),
            'hsv': (28, 27, 100),
            'ranges': {
                'hsv_lower': (20, 15, 85),
                'hsv_upper': (35, 40, 100)
            }
        },
        'DarkOrange': {
            'code': 'YC-226', 
            'name': 'Deep Orange',
            'lab': (60.0, 45.0, 70.0),
            'rgb': (255, 140, 0),
            'hsv': (33, 100, 100),
            'ranges': {
                'hsv_lower': (20, 80, 80),
                'hsv_upper': (40, 100, 100)
            }
        },
        
        # ======== PURPLES ========
        'Purple': {
            'code': 'YC-008', 
            'name': 'Royal Purple',
            'lab': (30.0, 48.0, -52.0),
            'rgb': (128, 0, 128),
            'hsv': (300, 100, 50),
            'ranges': {
                'hsv_lower': (280, 60, 30),
                'hsv_upper': (320, 100, 70)
            }
        },
        'Lavender': {
            'code': 'YC-018', 
            'name': 'Soft Lavender',
            'lab': (89.0, 12.0, -20.0),
            'rgb': (230, 230, 250),
            'hsv': (240, 8, 98),
            'ranges': {
                'hsv_lower': (230, 0, 80),
                'hsv_upper': (260, 20, 100)
            }
        },
        'Magenta': {
            'code': 'YC-021', 
            'name': 'Bright Magenta',
            'lab': (60.0, 92.0, -42.0),
            'rgb': (255, 0, 255),
            'hsv': (300, 100, 100),
            'ranges': {
                'hsv_lower': (290, 80, 80),
                'hsv_upper': (310, 100, 100)
            }
        },
        'Violet': {
            'code': 'YC-227', 
            'name': 'Violet',
            'lab': (40.0, 60.0, -60.0),
            'rgb': (138, 43, 226),
            'hsv': (271, 81, 89),
            'ranges': {
                'hsv_lower': (260, 60, 70),
                'hsv_upper': (280, 100, 100)
            }
        },
        'Indigo': {
            'code': 'YC-228', 
            'name': 'Indigo',
            'lab': (25.0, 25.0, -65.0),
            'rgb': (75, 0, 130),
            'hsv': (275, 100, 51),
            'ranges': {
                'hsv_lower': (265, 80, 30),
                'hsv_upper': (285, 100, 70)
            }
        },
        'Plum': {
            'code': 'YC-229', 
            'name': 'Plum',
            'lab': (40.0, 40.0, -20.0),
            'rgb': (142, 69, 133),
            'hsv': (307, 51, 56),
            'ranges': {
                'hsv_lower': (290, 40, 40),
                'hsv_upper': (315, 70, 70)
            }
        },
        
        # ======== NEUTRALS ========
        'White': {
            'code': 'YC-001', 
            'name': 'Pure White',
            'lab': (95.0, 0.0, 2.0),
            'rgb': (255, 255, 255),
            'hsv': (0, 0, 100),
            'ranges': {
                'hsv_lower': (0, 0, 90),
                'hsv_upper': (360, 10, 100)
            }
        },
        'Black': {
            'code': 'YC-002', 
            'name': 'Jet Black',
            'lab': (12.0, 0.0, 0.0),
            'rgb': (0, 0, 0),
            'hsv': (0, 0, 0),
            'ranges': {
                'hsv_lower': (0, 0, 0),
                'hsv_upper': (360, 30, 30)
            }
        },
        'Gray': {
            'code': 'YC-011', 
            'name': 'Stone Gray',
            'lab': (55.0, 0.0, 0.0),
            'rgb': (128, 128, 128),
            'hsv': (0, 0, 50),
            'ranges': {
                'hsv_lower': (0, 0, 30),
                'hsv_upper': (360, 15, 80)
            }
        },
        'Silver': {
            'code': 'YC-025', 
            'name': 'Metallic Silver',
            'lab': (80.0, 0.0, 0.0),
            'rgb': (192, 192, 192),
            'hsv': (0, 0, 75),
            'ranges': {
                'hsv_lower': (0, 0, 65),
                'hsv_upper': (360, 10, 85)
            }
        },
        'Charcoal': {
            'code': 'YC-230', 
            'name': 'Charcoal Gray',
            'lab': (30.0, 0.0, 0.0),
            'rgb': (54, 69, 79),
            'hsv': (210, 32, 31),
            'ranges': {
                'hsv_lower': (0, 0, 20),
                'hsv_upper': (360, 30, 40)
            }
        },
        'Taupe': {
            'code': 'YC-231', 
            'name': 'Taupe Gray',
            'lab': (60.0, 5.0, 15.0),
            'rgb': (139, 133, 137),
            'hsv': (315, 4, 55),
            'ranges': {
                'hsv_lower': (0, 0, 45),
                'hsv_upper': (360, 20, 65)
            }
        }
    }
    
    @classmethod
    def get_color_info(cls, color_name):
        """Get color information for the specified color name"""
        return cls.COLORS.get(color_name, None)
    
    @classmethod
    def get_basic_color_names(cls):
        """Get list of basic 8 color names"""
        return list(cls.BASIC_COLORS.keys())
    
    @classmethod
    def get_all_color_names(cls):
        """Get list of all color names"""
        return list(cls.COLORS.keys())

class YarnColorDetector:
    """Advanced yarn color detector with multiple detection methods"""
    
    def __init__(self):
        self.db = YarnColorDatabase
        self.basic_colors = self.db.get_basic_color_names()
        self.all_colors = self.db.get_all_color_names()
        self.color_history = deque(maxlen=20)  # Store recent color detections
        self.stable_count = {}  # Track stability of detections
        self.confidence_history = deque(maxlen=10)  # Track detection confidence
        
    def analyze_frame(self, frame):
        """Main analysis method for a camera frame - more responsive"""
        # Get the Region of Interest
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        region_size = min(180, min(h, w) // 3)  # Adapt to frame size
        
        x1 = max(0, center_x - region_size // 2)
        y1 = max(0, center_y - region_size // 2)
        x2 = min(w, x1 + region_size)
        y2 = min(h, y1 + region_size)
        
        # Define detection area
        detection_area = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width': x2-x1, 'height': y2-y1
        }
        
        # Extract the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Analyze the color in the ROI
        analysis = self._analyze_roi_color(roi)
        
        if analysis:
            # Update detection history
            self._update_detection_history(analysis)
            
            # Get most stable color over time - only apply for non-basic colors
            # or when confidence is low
            stable_color, stable_confidence = self._get_stable_color()
            
            if stable_color and (
                not analysis['is_basic'] or  # Always stabilize non-basic colors
                analysis['confidence'] < 0.7 or  # Stabilize low-confidence detections
                (stable_color == analysis['color_name'] and stable_confidence > analysis['confidence'])  # Same color but higher confidence
            ):
                analysis['color_name'] = stable_color
                analysis['confidence'] = stable_confidence
                analysis['method_used'] = 'STABLE_DETECTION'
                analysis['is_basic'] = stable_color in self.basic_colors
                # Update yarn code and name
                color_info = self.db.get_color_info(stable_color)
                if color_info:
                    analysis['yarn_code'] = color_info['code']
                    analysis['yarn_name'] = color_info['name']
        
        return analysis, detection_area

    
    def _analyze_roi_color(self, roi):
        """Analyze color in the region of interest using multiple methods"""
        if roi is None or roi.size == 0:
            return None
            
        # Extract dominant colors using different methods
        dominant_colors = self._extract_multiple_colors(roi)
        
        # ADDED: Gray Detection - Check this first before other methods
        r, g, b = dominant_colors['rgb_color']
        h, s, v = dominant_colors['hsv_color']
        
        # Calculate how similar R, G, and B values are to each other
        avg_val = (r + g + b) / 3
        color_variance = (abs(r - avg_val) + abs(g - avg_val) + abs(b - avg_val)) / 3
        
        # If saturation is low and RGB values are similar, it's probably gray
        if s < 20 and color_variance < 12:
            # Determine gray shade based on brightness
            if v > 85:
                color_name = "White"
            elif v > 60:
                color_name = "LightGray"
            elif v > 35:
                color_name = "Gray"
            else:
                color_name = "DarkGray"
                
            # Get color info
            color_info = self.db.get_color_info(color_name)
            if not color_info:
                # Default fallback if color isn't in database
                color_info = {'code': 'YC-GRY', 'name': color_name}
                
            # Return gray detection result with high confidence
            return {
                'color_name': color_name,
                'confidence': 0.95,
                'method_used': 'GRAY_DETECTION',
                'rgb_color': dominant_colors['rgb_color'],
                'hsv_color': dominant_colors['hsv_color'],
                'lab_values': dominant_colors['lab_values'],
                'lab_interpretation': dominant_colors['lab_interpretation'],
                'yarn_code': color_info['code'],
                'yarn_name': color_info['name'],
                'is_basic': True  # Consider grays as basic colors
            }
        
        # Analyze colors using multiple methods
        results = []
        
        # Method 0: Special direct RGB checks for problematic colors (HIGHEST PRIORITY)
        direct_detection = self._direct_color_checks(dominant_colors['rgb_color'])
        if direct_detection:
            results.append({
                'color_name': direct_detection[0],
                'confidence': direct_detection[1],
                'method': 'DIRECT_RGB_MATCH',
                'is_basic': direct_detection[0] in self.basic_colors
            })
        
        # Method 1: HSV range-based detection
        # First try with basic colors only
        hsv_basic_detection = self._detect_color_by_hsv_range(dominant_colors['hsv_color'], basic_only=True)
        if hsv_basic_detection:
            results.append({
                'color_name': hsv_basic_detection[0],
                'confidence': hsv_basic_detection[1],
                'method': 'HSV_BASIC',
                'is_basic': True
            })
        
        # Then try with all colors
        hsv_detection = self._detect_color_by_hsv_range(dominant_colors['hsv_color'], basic_only=False)
        if hsv_detection and (not hsv_basic_detection or hsv_detection[0] != hsv_basic_detection[0]):
            results.append({
                'color_name': hsv_detection[0],
                'confidence': hsv_detection[1] * 0.95,  # Slightly lower confidence for non-basic colors
                'method': 'HSV_RANGE',
                'is_basic': hsv_detection[0] in self.basic_colors
            })
        
        # Method 2: Special cream detection
        cream_detection = self._detect_cream_color(dominant_colors['rgb_color'])
        if cream_detection:
            results.append({
                'color_name': cream_detection[0],
                'confidence': cream_detection[1],
                'method': 'CREAM_SPECIAL',
                'is_basic': cream_detection[0] in self.basic_colors
            })
        
        # Method 3: LAB distance
        # First try with basic colors only
        lab_basic_detection = self._detect_color_by_lab(dominant_colors['lab_values'], basic_only=True)
        if lab_basic_detection:
            results.append({
                'color_name': lab_basic_detection[0],
                'confidence': lab_basic_detection[1] * 0.9,  # Slightly lower confidence for LAB
                'method': 'LAB_BASIC',
                'is_basic': True
            })
            
        # Then try with all colors
        lab_detection = self._detect_color_by_lab(dominant_colors['lab_values'], basic_only=False)
        if lab_detection and (not lab_basic_detection or lab_detection[0] != lab_basic_detection[0]):
            results.append({
                'color_name': lab_detection[0],
                'confidence': lab_detection[1] * 0.85,  # Lower confidence for non-basic colors with LAB
                'method': 'LAB_DISTANCE',
                'is_basic': lab_detection[0] in self.basic_colors
            })
        
        # Method 4: RGB similarity - lowest priority
        rgb_detection = self._detect_color_by_rgb(dominant_colors['rgb_color'])
        if rgb_detection:
            results.append({
                'color_name': rgb_detection[0],
                'confidence': rgb_detection[1] * 0.8,  # Lower confidence for RGB
                'method': 'RGB_SIMILARITY',
                'is_basic': rgb_detection[0] in self.basic_colors
            })
        
        # Prioritize results by basic colors and confidence
        if results:
            # Sort by 1) basic color status, 2) confidence
            results.sort(key=lambda x: (-1 if x['is_basic'] else 0, x['confidence']), reverse=True)
            
            # Pick the highest priority result
            best_result = results[0]
            
            # Get full color information
            color_info = self.db.get_color_info(best_result['color_name'])
            
            # Create complete analysis result
            return {
                'color_name': best_result['color_name'],
                'confidence': best_result['confidence'],
                'method_used': best_result['method'],
                'rgb_color': dominant_colors['rgb_color'],
                'hsv_color': dominant_colors['hsv_color'],
                'lab_values': dominant_colors['lab_values'],
                'lab_interpretation': dominant_colors['lab_interpretation'],
                'yarn_code': color_info['code'],
                'yarn_name': color_info['name'],
                'is_basic': best_result['is_basic']
            }
            
        return None
    
    def _extract_multiple_colors(self, roi):
        """Extract dominant colors from ROI using multiple techniques"""
        # Convert to different color spaces for analysis
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Method 1: K-means clustering to find dominant color
        reshaped = roi_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(reshaped, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels to find most common color clusters
        counts = np.bincount(labels.flatten())
        
        # Score each color by saturation and count
        best_score = -1
        dominant_color = None
        
        for i, center in enumerate(centers):
            # Convert to HSV to get saturation
            color_normalized = center / 255.0
            h, s, v = colorsys.rgb_to_hsv(color_normalized[0], color_normalized[1], color_normalized[2])
            
            # Calculate standard deviation (color variance)
            std_dev = np.std(center)
            
            # Calculate score based on saturation, variance and cluster size
            # Avoid very dark colors
            if v < 0.1:  # Too dark
                color_score = 0.1 * counts[i]
            # Special handling for cream/light colors
            elif v > 0.9 and s < 0.15:  # Potential cream
                color_score = 2.0 * counts[i]  # Boost cream detection
            # Special handling for yellow/orange
            elif 0.08 < h < 0.15 and s > 0.5 and v > 0.7:  # Orange hue range
                color_score = 2.5 * counts[i]  # Boost orange detection
            elif 0.15 < h < 0.20 and s > 0.5 and v > 0.7:  # Yellow hue range
                color_score = 2.5 * counts[i]  # Boost yellow detection
            # Normal colors
            else:
                color_score = (s * 2.0 + std_dev/40.0) * counts[i]
            
            if color_score > best_score:
                best_score = color_score
                dominant_color = center
        
        if dominant_color is None:
            # Fallback to median color
            dominant_color = np.median(roi_rgb.reshape(-1, 3), axis=0)
        
        # Convert to proper numpy array and dtype
        rgb_color = np.array(dominant_color, dtype=np.uint8)
        
        # Convert to HSV (normalized 0-1)
        r, g, b = rgb_color / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_color = (h, s, v)
        
        # Convert to LAB
        lab_color = self._rgb_to_lab(rgb_color)
        
        # Get LAB interpretation
        lab_interpretation = self._interpret_lab_values(lab_color[0], lab_color[1], lab_color[2])
        
        return {
            'rgb_color': rgb_color,
            'hsv_color': hsv_color,
            'lab_values': lab_color,
            'lab_interpretation': lab_interpretation
        }
    
    def _direct_color_checks(self, rgb):
        """Direct RGB checks for problematic colors like orange, yellow, cream"""
        r, g, b = map(int, rgb)  # Convert to signed integers to prevent overflow
        
        # Direct check for CREAM first - very high values with low variation
        if (r > 220 and g > 220 and b > 180 and 
            max(r, g, b) - min(r, g, b) < 40):
            return ('Cream', 0.97)
        
        # Direct check for YELLOW - high R & G, low B
        if ('Yellow' in self.basic_colors and
            r > 180 and g > 150 and b < 120 and
            abs(r - g) < 50 and (r - b) > 80):
            return ('Yellow', 0.97)
        
        # Direct check for ORANGE - high R, medium G, low B
        if ('Orange' in self.basic_colors and
            r > 180 and 70 < g < 180 and b < 100 and
            r > g * 1.3):  # R significantly higher than G
            return ('Orange', 0.97)
            
        # Check all basic colors with rgb_checks
        for color_name in self.basic_colors:
            color_info = self.db.get_color_info(color_name)
            if 'rgb_checks' in color_info:
                checks = color_info['rgb_checks']
                match = True
                
                # Check all specified conditions
                if 'min_r' in checks and r < checks['min_r']:
                    match = False
                if 'min_g' in checks and g < checks['min_g']:
                    match = False
                if 'min_b' in checks and b < checks['min_b']:
                    match = False
                if 'max_r' in checks and r > checks['max_r']:
                    match = False
                if 'max_g' in checks and g > checks['max_g']:
                    match = False
                if 'max_b' in checks and b > checks['max_b']:
                    match = False
                if 'r_g_ratio' in checks and r < g * checks['r_g_ratio']:
                    match = False
                if 'r_g_similarity' in checks and abs(r - g) > checks['r_g_similarity']:
                    match = False
                if 'r_b_diff' in checks and (r - b) < checks['r_b_diff']:
                    match = False
                
                if match:
                    return (color_name, 0.96)
        
        return None
    
    def _detect_color_by_hsv_range(self, hsv, basic_only=False):
        """Detect color by checking if it falls within HSV ranges"""
        h, s, v = hsv
        h_scaled = h * 360  # Convert to 0-360 range
        s_scaled = s * 100  # Convert to percentage
        v_scaled = v * 100  # Convert to percentage
        
        matches = []
        
        # Create dictionary of colors to check
        colors_to_check = self.db.BASIC_COLORS if basic_only else self.db.COLORS
        
        # Check all colors with defined HSV ranges
        for color_name, color_data in colors_to_check.items():
            if 'ranges' in color_data:
                hsv_lower = color_data['ranges']['hsv_lower']
                hsv_upper = color_data['ranges']['hsv_upper']
                
                # Handle hue wrap-around
                in_hue_range = False
                if hsv_lower[0] > hsv_upper[0]:  # Wrap around 0/360
                    in_hue_range = (h_scaled >= hsv_lower[0] or h_scaled <= hsv_upper[0])
                else:
                    in_hue_range = (hsv_lower[0] <= h_scaled <= hsv_upper[0])
                
                in_s_range = (hsv_lower[1] <= s_scaled <= hsv_upper[1])
                in_v_range = (hsv_lower[2] <= v_scaled <= hsv_upper[2])
                
                if in_hue_range and in_s_range and in_v_range:
                    # Calculate how centered in range (better = higher confidence)
                    h_center = min(abs(h_scaled - (hsv_lower[0] + hsv_upper[0])/2), 
                                   abs(h_scaled - (hsv_lower[0] + hsv_upper[0] - 360)/2))
                    s_center = abs(s_scaled - (hsv_lower[1] + hsv_upper[1])/2)
                    v_center = abs(v_scaled - (hsv_lower[2] + hsv_upper[2])/2)
                    
                    # Normalize distances
                    h_norm = h_center / 180.0  # Half of hue range
                    s_norm = s_center / 50.0   # Half of saturation range
                    v_norm = v_center / 50.0   # Half of value range
                    
                    # Weighted score - lower is better
                    range_score = h_norm * 0.6 + s_norm * 0.3 + v_norm * 0.1
                    
                    # Convert to confidence (higher is better)
                    confidence = max(0.0, min(0.99, 1.0 - range_score))
                    
                    # Boost confidence for basic colors
                    if color_name in self.basic_colors:
                        confidence = min(0.99, confidence * 1.1)
                    
                    # Special boost for problematic colors
                    if color_name in ['Yellow', 'Orange', 'Cream']:
                        confidence = min(0.99, confidence * 1.1)
                    
                    # Add to matches
                    matches.append((color_name, confidence))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match if any
        if matches:
            return matches[0]
        
        return None
    
    def _detect_cream_color(self, rgb):
        """Special detection for cream/beige colors which are hard to detect"""
        r, g, b = rgb
        
        # Check for cream colors using RGB thresholds
        for color_name, color_data in self.db.COLORS.items():
            if 'rgb_thresholds' in color_data:
                thresholds = color_data['rgb_thresholds']
                
                # Check if color meets thresholds
                if (r >= thresholds['min_r'] and 
                    g >= thresholds['min_g'] and 
                    b >= thresholds['min_b'] and
                    max(r, g, b) - min(r, g, b) <= thresholds['max_diff']):
                    
                    # Calculate similarity to ideal cream
                    ref_r, ref_g, ref_b = color_data['rgb']
                    
                    # Distance calculation
                    distance = np.sqrt(
                        (r - ref_r)**2 + 
                        (g - ref_g)**2 + 
                        (b - ref_b)**2
                    ) / 255.0  # Normalize to 0-1
                    
                    # Convert distance to confidence
                    confidence = max(0.7, min(0.98, 1.0 - distance))
                    
                    # Always give cream a high confidence if it matches thresholds
                    if color_name == 'Cream':
                        confidence = max(confidence, 0.92)
                    
                    return (color_name, confidence)
        
        return None
    
    def _detect_color_by_lab(self, lab, basic_only=False):
        """Detect color using LAB color distance"""
        L, a, b = lab
        
        # Weights for LAB components
        L_weight = 0.5   # Lightness less important
        a_weight = 1.5   # Red-green axis more important
        b_weight = 1.2   # Yellow-blue axis important
        
        min_distance = float('inf')
        closest_color = None
        
        # Create dictionary of colors to check
        colors_to_check = self.db.BASIC_COLORS if basic_only else self.db.COLORS
        
        # Calculate distance to all colors
        for color_name, color_data in colors_to_check.items():
            ref_L, ref_a, ref_b = color_data['lab']
            
            # Calculate weighted Euclidean distance
            distance = np.sqrt(
                (L_weight * (L - ref_L))**2 + 
                (a_weight * (a - ref_a))**2 + 
                (b_weight * (b - ref_b))**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        if closest_color:
            # Convert distance to confidence
            confidence = max(0.0, min(0.95, 1.0 - min_distance/100.0))
            
            # Boost confidence for basic colors
            if closest_color in self.basic_colors:
                confidence = min(0.95, confidence * 1.1)
                
            return (closest_color, confidence)
        
        return None
    
    def _detect_color_by_rgb(self, rgb):
        """Detect color using RGB similarity"""
        min_distance = float('inf')
        closest_color = None
        closest_basic = None
        min_basic_distance = float('inf')
        
        # Calculate distance to all colors
        for color_name, color_data in self.db.COLORS.items():
            ref_rgb = color_data['rgb']
            
            # Calculate Euclidean distance in RGB space
            distance = np.sqrt(
                (rgb[0] - ref_rgb[0])**2 + 
                (rgb[1] - ref_rgb[1])**2 + 
                (rgb[2] - ref_rgb[2])**2
            )
            
            # Track closest overall and closest basic color
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
                
            if color_name in self.basic_colors and distance < min_basic_distance:
                min_basic_distance = distance
                closest_basic = color_name
        
        # Prefer basic colors unless non-basic is significantly closer
        if closest_basic and min_basic_distance < min_distance * 1.3:
            closest_color = closest_basic
            min_distance = min_basic_distance
        
        if closest_color:
            # Convert distance to confidence
            confidence = max(0.0, min(0.9, 1.0 - min_distance/255.0))
            
            # Boost confidence for basic colors
            if closest_color in self.basic_colors:
                confidence = min(0.9, confidence * 1.1)
                
            return (closest_color, confidence)
        
        return None
    
    def _update_detection_history(self, analysis):
        """Update detection history for stability analysis - with less inertia"""
        if not analysis:
            return
            
        color_name = analysis['color_name']
        
        # Add to history
        self.color_history.append(color_name)
        
        # Update stable count with MORE AGGRESSIVE updates for basic colors
        if color_name not in self.stable_count:
            # New colors get a higher initial value if they're basic colors
            self.stable_count[color_name] = 2.0 if color_name in self.basic_colors else 1.0
        else:
            # Basic colors accumulate stability faster
            if color_name in self.basic_colors:
                self.stable_count[color_name] += 1.5
            else:
                self.stable_count[color_name] += 1.0
        
        # Decay counts for unused colors FASTER
        for c in list(self.stable_count.keys()):
            if c != color_name:
                # Faster decay for non-basic colors
                decay_rate = 0.5 if c in self.basic_colors else 0.8
                self.stable_count[c] = max(0, self.stable_count[c] - decay_rate)
                # Remove colors with zero count
                if self.stable_count[c] <= 0:
                    del self.stable_count[c]
        
        # Update confidence
        self.confidence_history.append(analysis['confidence'])

    def _direct_color_checks(self, rgb):
        """Direct RGB checks for problematic colors like orange, yellow, cream"""
        r, g, b = map(int, rgb)  # Convert to signed integers to prevent overflow
        
        # Direct check for CREAM first - very high values with low variation
        if (r > 220 and g > 220 and b > 180 and 
            max(r, g, b) - min(r, g, b) < 40):
            return ('Cream', 0.97)
        
        # Direct check for YELLOW - high R & G, low B
        if ('Yellow' in self.basic_colors and
            r > 180 and g > 150 and b < 120 and
            abs(r - g) < 50 and (r - b) > 80):
            return ('Yellow', 0.97)
        
        # Direct check for ORANGE - high R, medium G, low B
        if ('Orange' in self.basic_colors and
            r > 180 and 70 < g < 180 and b < 100 and
            r > g * 1.3):  # R significantly higher than G
            return ('Orange', 0.97)
            
        # Check all basic colors with rgb_checks
        for color_name in self.basic_colors:
            color_info = self.db.get_color_info(color_name)
            if 'rgb_checks' in color_info:
                checks = color_info['rgb_checks']
                match = True
                
                # Check all specified conditions
                if 'min_r' in checks and r < checks['min_r']:
                    match = False
                if 'min_g' in checks and g < checks['min_g']:
                    match = False
                if 'min_b' in checks and b < checks['min_b']:
                    match = False
                if 'max_r' in checks and r > checks['max_r']:
                    match = False
                if 'max_g' in checks and g > checks['max_g']:
                    match = False
                if 'max_b' in checks and b > checks['max_b']:
                    match = False
                if 'r_g_ratio' in checks and r < g * checks['r_g_ratio']:
                    match = False
                if 'r_g_similarity' in checks and abs(r - g) > checks['r_g_similarity']:
                    match = False
                if 'r_b_diff' in checks and (r - b) < checks['r_b_diff']:
                    match = False
                
                if match:
                    return (color_name, 0.96)
        
        return None

    def _get_stable_color(self):
        """Get the most stable color detection over time - more responsive"""
        if not self.color_history or len(self.color_history) == 0:
            return None, 0.0
        
        # Convert color_history to list manually for safety
        color_history_list = []
        for color in self.color_history:
            color_history_list.append(color)
        
        # Get recent colors (last 5 or fewer)
        recent_count = min(5, len(color_history_list))
        recent_colors = color_history_list[-recent_count:]
        
        # Count occurrences in recent history with high weighting for most recent
        weighted_counts = {}
        for i in range(len(recent_colors)):
            color = recent_colors[i]
            # Much higher weight for the most recent color
            weight = 0.5 + 0.5 * (i / len(recent_colors))
            weight = weight * 1.5 if i == len(recent_colors) - 1 else weight
            
            # Extra weight for basic colors
            if color in self.basic_colors:
                weight *= 1.2
                
            if color in weighted_counts:
                weighted_counts[color] += weight
            else:
                weighted_counts[color] = weight
        
        # Find most common color
        if not weighted_counts:
            return None, 0.0
        
        most_common_color = None
        highest_count = -1
        for color, count in weighted_counts.items():
            if count > highest_count:
                most_common_color = color
                highest_count = count
        
        if most_common_color is None:
            return None, 0.0
        
        # If we have at least one color in recent history
        if len(recent_colors) > 0:
            most_recent = recent_colors[-1]
            
            # If the most recent detection is a basic color, use it immediately
            # unless another basic color has much higher stability
            if most_recent in self.basic_colors:
                use_most_recent = True
                for color, count in self.stable_count.items():
                    if (color in self.basic_colors and 
                        color != most_recent and 
                        count > self.stable_count.get(most_recent, 0) * 1.5):
                        # Another basic color has much higher stability
                        most_common_color = color
                        use_most_recent = False
                        break
                
                if use_most_recent:
                    most_common_color = most_recent
        
        # Calculate confidence based on stability and detection confidence
        stability = min(0.99, weighted_counts.get(most_common_color, 0) / 5.0)
        
        # Calculate average confidence safely
        confidence_list = []
        for conf in self.confidence_history:
            confidence_list.append(conf)
        
        if confidence_list:
            avg_confidence = sum(confidence_list) / len(confidence_list)
        else:
            avg_confidence = 0.7
        
        confidence = 0.6 * stability + 0.4 * avg_confidence
        
        # Boost confidence for basic colors
        if most_common_color in self.basic_colors:
            confidence = min(0.99, confidence * 1.05)
        
        return most_common_color, confidence

    
    def _rgb_to_lab(self, rgb):
        """Convert RGB color to LAB"""
        try:
            # Create a single-pixel image with this color
            pixel = np.uint8([[rgb]])
            
            # Convert RGB to LAB
            pixel_bgr = cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR)
            pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)
            L, a, b = pixel_lab[0, 0]
            
            # Convert to standard LAB range
            L_star = float(L) * 100.0 / 255.0
            a_star = float(a) - 128.0
            b_star = float(b) - 128.0
            
            return (L_star, a_star, b_star)
        except Exception:
            return (50.0, 0.0, 0.0)  # Default safe value
    
    def _interpret_lab_values(self, L, a, b):
        """Interpret LAB values into descriptive terms"""
        # Brightness interpretation
        if L > 85: brightness = "Very Bright"
        elif L > 70: brightness = "Bright"
        elif L > 50: brightness = "Medium"
        elif L > 30: brightness = "Dark"
        else: brightness = "Very Dark"
        
        # Red-Green axis
        if a > 40: red_green = "Strongly Reddish"
        elif a > 15: red_green = "Reddish"
        elif a < -15: red_green = "Greenish"
        else: red_green = "Neutral"
        
        # Yellow-Blue axis - ADJUST THESE THRESHOLDS
        if b > 25: yellow_blue = "Strongly Yellowish"
        elif b > 10: yellow_blue = "Yellowish" 
        elif b < -25: yellow_blue = "Strongly Bluish"
        elif b < -10: yellow_blue = "Bluish"
        else: yellow_blue = "Neutral"
        
        return (brightness, red_green, yellow_blue)
    
    def get_color_display_value(self, color_name):
        """Get BGR value for display purposes"""
        color_map = {
            'White': (250, 250, 250), 'Black': (20, 20, 20),
            'Red': (40, 40, 220), 'Blue': (220, 120, 40),
            'Green': (40, 220, 40), 'Yellow': (40, 250, 250),
            'Orange': (40, 130, 250), 'Purple': (220, 120, 220),
            'Pink': (130, 130, 250), 'Brown': (20, 50, 120),
            'Gray': (120, 120, 120), 'Cream': (220, 240, 250),
            'Navy': (120, 0, 0), 'Maroon': (0, 0, 120),
            'Turquoise': (190, 210, 50), 'Lime': (30, 220, 30),
            'Coral': (70, 110, 250), 'Lavender': (250, 220, 220),
            'Teal': (120, 120, 0), 'Gold': (0, 190, 250),
            'Magenta': (250, 0, 250), 'Beige': (200, 230, 230),
            'Olive': (0, 120, 120), 'Crimson': (50, 10, 220),
            'Silver': (190, 190, 190), 'Aqua': (250, 250, 0),
            'Indigo': (130, 0, 75),
            # Extended colors
            'DarkBrown': (20, 30, 90), 'LightBrown': (40, 100, 180),
            'BrightRed': (0, 0, 255), 'DarkRed': (0, 0, 100),
            'LightPink': (180, 180, 255), 'DarkGreen': (0, 100, 0),
            'LightBlue': (240, 200, 140), 'LightCream': (240, 250, 255),
            'Tan': (140, 180, 210), 'Peach': (185, 218, 255),
            'SkyBlue': (235, 206, 135), 'Mint': (152, 255, 152),
            'Amber': (0, 191, 255), 'Plum': (133, 69, 142),
            'Violet': (226, 43, 138), 'Ivory': (240, 255, 255)
        }
        return color_map.get(color_name, (100, 100, 100))

class YarnDetectorUI:
    """Enhanced UI for the yarn color detector"""
    
    def __init__(self, width=1000, height=700):
        self.width = width
        self.height = height
        self.display_frame = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_height = 22
        
        # Create empty frame
        self.reset_display()
    
    def reset_display(self):
        """Reset display frame"""
        self.display_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def create_info_panel(self, analysis_result, fps=0):
        """Create information panel with detection results"""
        panel_height = 280
        info_panel = np.zeros((panel_height, self.width, 3), dtype=np.uint8)
        
        # Draw background
        cv2.rectangle(info_panel, (0, 0), (self.width, panel_height), (20, 20, 20), -1)
        
        if analysis_result is None:
            cv2.putText(info_panel, "No Detection", (20, 50), 
                       self.font, 1.0, (100, 100, 100), 2)
            return info_panel
        
        # Extract data from analysis
        color_name = analysis_result['color_name']
        confidence = analysis_result.get('confidence', 0.7)
        method_used = analysis_result.get('method_used', 'UNKNOWN')
        L, a, b = analysis_result['lab_values']
        brightness, red_green, yellow_blue = analysis_result['lab_interpretation']
        yarn_code = analysis_result.get('yarn_code', 'YC-???')
        yarn_name = analysis_result.get('yarn_name', 'Unknown')
        is_basic = analysis_result.get('is_basic', False)
        
        # Bar visualization for confidence
        bar_width = int((self.width - 40) * confidence)
        bar_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.rectangle(info_panel, (20, 20), (20 + bar_width, 35), bar_color, -1)
        cv2.rectangle(info_panel, (20, 20), (self.width - 20, 35), (255, 255, 255), 1)
        
        # Main detection result
        header_color = (0, 255, 255) if is_basic else (200, 200, 200)
        cv2.putText(info_panel, f"DETECTED: {color_name} ({confidence*100:.1f}%)", 
                   (20, 60), self.font, 0.8, header_color, 2)
        
        # Method indicator
        method_colors = {
            'DIRECT_RGB_MATCH': (0, 255, 0),    # Green - Most reliable
            'HSV_BASIC': (0, 255, 0),           # Green - Best method
            'HSV_RANGE': (0, 255, 0),           # Green - Best method
            'CREAM_SPECIAL': (0, 255, 255),     # Cyan - Special cream detection
            'LAB_BASIC': (0, 200, 255),         # Light orange - Standard method
            'LAB_DISTANCE': (0, 165, 255),      # Orange - Standard method
            'RGB_SIMILARITY': (255, 165, 0),    # Blue - Less accurate
            'STABLE_DETECTION': (255, 255, 0)   # Yellow - Temporal stability
        }
        method_color = method_colors.get(method_used, (255, 255, 255))
        cv2.putText(info_panel, f"Method: {method_used}", 
                   (self.width - 240, 60), self.font, 0.45, method_color, 1)
        
        # Yarn code information
        cv2.putText(info_panel, f"CODE: {yarn_code} - {yarn_name}", 
                   (20, 85), self.font, 0.6, (150, 255, 150), 1)
        
        # Basic color indicator
        if is_basic:
            cv2.putText(info_panel, "BASIC COLOR", 
                       (self.width - 240, 30), self.font, 0.5, (0, 255, 255), 1)
        
        # CIELAB values
        cv2.putText(info_panel, f"L* = {L:.1f} ({brightness})", 
                (20, 125), self.font, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, f"a* = {a:.1f} ({red_green})", 
                (20, 145), self.font, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, f"b* = {b:.1f} ({yellow_blue})", 
                (20, 165), self.font, 0.5, (200, 200, 200), 1)
        
        # RGB values
        rgb = analysis_result['rgb_color']
        cv2.putText(info_panel, f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})", 
                (20, 250), self.font, 0.45, (200, 200, 200), 1)   
        
        # FPS indicator
        cv2.putText(info_panel, f"FPS: {fps:.1f}", 
                (self.width - 120, 250), self.font, 0.5, (200, 200, 200), 1)
        
        # Color indicator circle
        detector = YarnColorDetector()
        display_color = detector.get_color_display_value(color_name)
        cv2.circle(info_panel, (self.width - 60, 60), 35, display_color, -1)
        cv2.circle(info_panel, (self.width - 60, 60), 35, (255, 255, 255), 2)
        
        # Command help text
        cv2.putText(info_panel, "SPACE=capture  'q'=exit", 
                (20, 270), self.font, 0.4, (150, 150, 150), 1)
        
        return info_panel
    
    def create_frame_with_detection(self, frame, analysis_result, detection_area, fps=0):
        """Create a complete display frame with detection area and info panel"""
        # Create a copy of the frame
        display = frame.copy()
        
        # Draw detection area if provided
        if detection_area:
            x1, y1, x2, y2 = detection_area['x1'], detection_area['y1'], detection_area['x2'], detection_area['y2']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "DETECTION AREA", (x1, y1-10), 
                    self.font, 0.6, (0, 255, 0), 2)
        
        # Create info panel
        info_panel = self.create_info_panel(analysis_result, fps)
        
        # Resize frame if needed to match width
        h, w = display.shape[:2]
        if w != self.width:
            aspect = h / w
            new_h = int(self.width * aspect)
            display = cv2.resize(display, (self.width, new_h))
        
        # Combine frame and info panel
        self.display_frame = np.vstack([display, info_panel])
        
        return self.display_frame
    
    def create_stats_overlay(self, color_history, detection_count=0):
        """Create statistics overlay"""
        if not color_history:
            return None
            
        # Count colors
        color_counts = Counter(color_history)
        
        # Create overlay
        overlay = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (400, 300), (40, 40, 40), -1)
        cv2.rectangle(overlay, (0, 0), (400, 300), (255, 255, 255), 1)
        
        # Title
        cv2.putText(overlay, "COLOR DETECTION STATISTICS", (20, 30), 
                self.font, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"Total Detections: {len(color_history)}", (20, 55), 
                self.font, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"Captures Saved: {detection_count}", (20, 75), 
                self.font, 0.5, (200, 200, 200), 1)
        
        # Color distribution
        y_offset = 110
        cv2.putText(overlay, "RECENT COLOR DISTRIBUTION:", (20, y_offset - 10), 
                self.font, 0.4, (150, 150, 150), 1)
                
        # Sort colors by count
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        
        detector = YarnColorDetector()
        for i, (color, count) in enumerate(sorted_colors[:8]):  # Show top 8
            percentage = count / len(color_history) * 100
            
            # Color indicator
            display_color = detector.get_color_display_value(color)
            cv2.rectangle(overlay, (20, y_offset + i*20), (40, y_offset + i*20 + 15), display_color, -1)
            cv2.rectangle(overlay, (20, y_offset + i*20), (40, y_offset + i*20 + 15), (255, 255, 255), 1)
            
            # Color name and count
            cv2.putText(overlay, f"{color}: {count} ({percentage:.1f}%)", 
                    (50, y_offset + i*20 + 12), self.font, 0.4, (200, 200, 200), 1)
        
        # Footer
        cv2.putText(overlay, "Press any key to close", (20, 280), 
                self.font, 0.4, (150, 150, 150), 1)
                
        return overlay
       
    def save_detection_result(analysis_result, frame, roi=None):
        """Save detection results with enhanced reporting"""
        if analysis_result is None:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        color_name = analysis_result['color_name']
        yarn_code = analysis_result.get('yarn_code', 'YC-000')
        method_used = analysis_result.get('method_used', 'UNKNOWN')
        L, a, b = analysis_result['lab_values']
        
        try:
            # Create directory if not exists
            os.makedirs('detections', exist_ok=True)
            
            # Save main image file
            filename = f"yarn_{timestamp}_{color_name}_{yarn_code}.jpg"
            filepath = os.path.join('detections', filename)
            cv2.imwrite(filepath, frame)
            
            # If ROI is provided, save it too
            if roi is not None and roi.size > 0:
                roi_path = os.path.join('detections', f"roi_{timestamp}_{color_name}.jpg")
                cv2.imwrite(roi_path, roi)
            
            # Create detailed analysis report
            report_path = os.path.join('detections', f"report_{timestamp}_{color_name}.txt")
            with open(report_path, 'w') as f:
                f.write("=============================================\n")
                f.write("YARN COLOR DETECTOR - DETECTION REPORT\n")
                f.write("=============================================\n\n")
                
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Filename: {filename}\n\n")
                
                f.write("COLOR INFORMATION\n")
                f.write("-----------------\n")
                f.write(f"Detected Color: {color_name}\n")
                f.write(f"Yarn Code: {yarn_code}\n")
                f.write(f"Yarn Name: {analysis_result.get('yarn_name', 'Unknown')}\n")
                f.write(f"Detection Method: {method_used}\n")
                f.write(f"Confidence: {analysis_result.get('confidence', 0.7)*100:.1f}%\n\n")
                
                f.write("COLOR VALUES\n")
                f.write("-----------------\n")
                f.write(f"L* (Lightness): {L:.2f}\n")
                f.write(f"a* (Red-Green): {a:.2f}\n")
                f.write(f"b* (Yellow-Blue): {b:.2f}\n\n")
                
                brightness, red_green, yellow_blue = analysis_result['lab_interpretation']
                f.write("COLOR INTERPRETATION\n")
                f.write("-----------------\n")
                f.write(f"Brightness: {brightness}\n")
                f.write(f"Red-Green Axis: {red_green}\n")
                f.write(f"Yellow-Blue Axis: {yellow_blue}\n\n")
                
                rgb = analysis_result['rgb_color']
                hsv = analysis_result['hsv_color']
                h, s, v = hsv
                
                f.write("ADDITIONAL COLOR DATA\n")
                f.write("-----------------\n")
                f.write(f"RGB Values: ({rgb[0]}, {rgb[1]}, {rgb[2]})\n")
                f.write(f"HSV Values: H:{h*360:.1f}° S:{s*100:.1f}% V:{v*100:.1f}%\n\n")
                
                # Add info about whether it's a basic color
                is_basic = analysis_result.get('is_basic', False)
                f.write(f"Basic Color: {'Yes' if is_basic else 'No'}\n\n")
                
                f.write("=============================================\n")
                f.write("Generated by YarnColorDetector\n")
            
            print(f"Detection saved as: {filename}")
            print(f"Report saved as: {os.path.basename(report_path)}")
            return True
        except Exception as e:
            print(f"Error saving detection: {e}")
            return False
        
if __name__ == "__main__":
    # Yarn Color Detector Main Program
    print("\nYARN COLOR DETECTOR - CAMERA MODE")
    print("================================")
    print("An enhanced application for real-time yarn color analysis")
    print("================================\n")
    
    # Initialize detector and UI
    detector = YarnColorDetector()
    ui = YarnDetectorUI(width=800, height=700)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Camera not available.")
        exit()
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set to minimal buffer size for responsiveness
    
    print("🎮 Controls: SPACE=capture, 'q'=exit")
    
    # Variables for statistics
    detection_count = 0
    color_history = []
    fps_history = deque(maxlen=30)  # For FPS smoothing
    last_time = time.time()
    
    # Frame counter for periodic camera reset
    frame_counter = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break
        
        # Reset camera buffer periodically to ensure fresh frames
        frame_counter += 1
        if frame_counter > 100:  # Reset every ~3 seconds at 30 FPS
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame_counter = 0
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        last_time = current_time
        
        # Analyze frame
        analysis, detection_area = detector.analyze_frame(frame)
        
        # Update color history if we have a result
        if analysis:
            color_history.append(analysis['color_name'])
            if len(color_history) > 100:  # Limit history size
                color_history.pop(0)
        
        # Create display frame
        display_frame = ui.create_frame_with_detection(
            frame, analysis, detection_area, avg_fps)
        
        # Show the frame
        cv2.imshow('Yarn Color Detector', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Capture
            if analysis:
                # Get ROI from detection area
                roi = None
                if detection_area:
                    x1, y1, x2, y2 = detection_area['x1'], detection_area['y1'], detection_area['x2'], detection_area['y2']
                    roi = frame[y1:y2, x1:x2]
                
            # Create stats overlay
            stats_overlay = ui.create_stats_overlay(color_history, detection_count)
            if stats_overlay is not None:
                # Center overlay on screen
                h, w = display_frame.shape[:2]
                overlay_h, overlay_w = stats_overlay.shape[:2]
                x_offset = (w - overlay_w) // 2
                y_offset = (h - overlay_h) // 2
                
                # Create copy of display frame
                stats_frame = display_frame.copy()
                
                # Apply semi-transparent background
                cv2.rectangle(stats_frame, 
                             (x_offset-10, y_offset-10), 
                             (x_offset+overlay_w+10, y_offset+overlay_h+10), 
                             (0, 0, 0), -1)
                
                # Apply overlay
                stats_frame[y_offset:y_offset+overlay_h, 
                           x_offset:x_offset+overlay_w] = stats_overlay
                
                # Show statistics
                cv2.imshow('Yarn Color Detector - Statistics', stats_frame)
                cv2.waitKey(0)  # Wait for any key
                cv2.destroyWindow('Yarn Color Detector - Statistics')
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
