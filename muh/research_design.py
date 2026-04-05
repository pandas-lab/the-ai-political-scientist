"""
Research Design: Historical Trauma and Cultural Consumption

HYPOTHESIS: Cities with anti-Japanese war memorials (indicating historical
trauma from Japanese invasion) will have lower audience counts for Demon Slayer
(a Japanese anime film) relative to Zootopia 2 (an American-Chinese animated film).

IDENTIFICATION STRATEGY: Difference-in-Differences

We exploit the staggered release of two animated films:
- Demon Slayer (Japanese): Nov 13 - Dec 11, 2025
- Zootopia 2 (American-Chinese): Nov 26 - Dec 26, 2025

Treatment: Cities with anti-Japanese war memorials
Control: Cities without such memorials

The key identifying assumption is that cities with memorials would have shown
similar relative preferences for Japanese vs. American content absent the
historical trauma captured by memorial presence.

We estimate:
  audience_count = β1*treatment + β2*japanese_film + β3*(treatment*japanese_film) + controls

Where β3 captures the differential effect of historical trauma on Japanese content consumption.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("RESEARCH DESIGN: Historical Trauma and Cultural Consumption")
print("="*60)
print()
print("HYPOTHESIS:")
print("Cities with anti-Japanese war memorials will show lower relative")
print("audience counts for Demon Slayer (Japanese) vs. Zootopia 2 (American).")
print()
print("IDENTIFICATION: Difference-in-Differences")
print("- Treatment: Cities with anti-Japanese memorials")
print("- Japanese film: Demon Slayer (treatment period)")
print("- American film: Zootopia 2 (control period)")
print("- Overlap period: Nov 26 - Dec 11, 2025")
print()
print("ESTIMAND: β3 in audience = β1*Treat + β2*Japanese + β3*Treat×Japanese")
print("="*60)
