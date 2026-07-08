"""Baseball-specific configuration constants."""

from __future__ import annotations

# Quirky-dimension MLB parks forced entirely into the test set for the baseball
# "park" split (a park-factor distribution shift that is harder than a random
# holdout and complementary to the chronological cross-season split). Identified
# by the Statcast home_team code: Fenway (BOS), Coors (COL), Yankee Stadium
# (NYY), Great American (CIN), Oracle (SF), Wrigley (CHC), and Minute Maid's
# Crawford Boxes (HOU). Adjust freely; HR rate by park spans ~0.030-0.061.
OUTLIER_BALLPARKS = ["BOS", "COL", "NYY", "CIN", "SF", "CHC", "HOU"]
