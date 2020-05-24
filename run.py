# Option Parser
from optparse import OptionParser

# Train and Test
import train
import test
import sys
#sys.stdout = open('./logs/resnet_1000_100.log', 'w')


parser = OptionParser()

parser.add_option("-m", "--mode", dest="mode", help="Mode (Train or Test)")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=1000)
parser.add_option("--epoch_length", type="int", dest="epoch_length", help="Length of epochs.", default=100)
parser.add_option("-l", "--learning_rate", type="float", dest="learning_rate", default="0.001")
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("-o", "--output_weight_path", dest="output_weight_path", help="Output path for weights.", default="./weights/model_frcnn.hdf5")
parser.add_option("-i", "--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.", default="./weights/model_frcnn.hdf5")

(options, args) = parser.parse_args()

if not options.mode:
    parser.error("Specify mode")

if options.mode == "train":
    print("Training")
    input_weight_path = options.input_weight_path
    output_weight_path = options.output_weight_path
    learning_rate = options.learning_rate
    epoch_length = options.epoch_length
    num_epochs = options.num_epochs
    train.train(input_weight_path, output_weight_path, learning_rate, epoch_length, num_epochs)

elif options.mode == "test":
    test.test()

else:
    parser.error("Invalid mode")

#sys.stdout.close()
