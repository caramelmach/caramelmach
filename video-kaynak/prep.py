import sys
from PIL import Image
import numpy as np
from rembg import remove, new_session
d = sys.argv[1]
s = new_session('u2net_human_seg')
s2 = new_session('isnet-general-use')
for n in ['hero', 'p62', 'p90', 'team_a', 'wantyou']:
    im = Image.open(f'{d}/{n}.png').convert('RGB')
    big = im.resize((im.width*2, im.height*2), Image.LANCZOS)
    out = remove(big, session=s if n != 'team_a' else s2)
    out.save(f'{d}/{n}_cut.png')
    print(n, out.size)
