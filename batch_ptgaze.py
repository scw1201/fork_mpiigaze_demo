#!/media/lenovo/本地磁盘/1_talkingface/removed_env_for_space/mpg_demo/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from ptgaze.batch_main import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())