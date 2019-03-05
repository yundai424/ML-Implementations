def get_config():
    return Config()


class Config(object):
    def __init__(self):
        self.keep_prob = .8
        self.face_path = "./face.npy"
        self.non_face_path = "./nonface.npy"
        self.model_path = "./trained_model/"
        self.model_name = "latest.ckpt"
        self.stride1 = 2
        self.stride2 = 2
        self.convsize1 = 3
        self.convsize2 = 3
        self.learning_rate = 0.01
        self.batch_size = 32
        self.epoch = 10
        self.test_ratio = 0.1
