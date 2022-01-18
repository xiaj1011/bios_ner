import logging


log = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s"))
log.addHandler(handler)
log.setLevel(logging.INFO)