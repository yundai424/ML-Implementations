class Config(object):
	def __init__(self):
		self.train_file = './bobsue-data/bobsue.seq2seq.train.tsv'
		self.validate_file = './bobsue-data/bobsue.seq2seq.dev.tsv'
		self.test_file = './bobsue-data/bobsue.seq2seq.test.tsv'
		self.all_voc_dict = './bobsue-data/bobsue.voc.txt'
		self.glove_dict = './glove.6B.200d.txt'
		self.embedding_dict = './pretrained_embedding.npy'
		self.lr = 0.5
		self.lr_decay = 0.8
		self.max_epoch = 25
		self.embedding = 'precomp'
		self.encoder_path = './encoder_final.pt'
		self.decoder_path = './decoder_final.pt'
		self.loss_plot_path = './loss.png'
		self.predict_result = './prediction_result.txt'
		
def get_config():
	return Config()