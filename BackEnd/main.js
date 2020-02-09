// Load in Pose Estimation Libraries
const tfjs = require('@tensorflow/tfjs-node-gpu');
tfjs.enableProdMode();
const posenet = require('@tensorflow-models/posenet');

// Load in Web Server Libraries
const service = require('restana')();
const bodyParser = require('body-parser');
const cors = require('cors')

// Load in Image Processing Libraries
const { createCanvas, loadImage, Image } = require('canvas');
const dataUriToBuffer = require('data-uri-to-buffer');
const sharp = require('sharp');

// Set global image resising dim
const imageDim = 200;


const networkSettings = {
	architecture: 'ResNet50',
	outputStride: 32,
	inputResolution: { width: imageDim, height: imageDim },
	quantBytes: 4,
}

var net = undefined;

const canvas = createCanvas(imageDim, imageDim);
const ctx = canvas.getContext('2d');

// Setup Webserver
service.use(cors());
service.use(bodyParser.json({limit: '50mb'}));

service.post('/', async (req, res) => {
	// Load Pose Estimation Network if not already loaded
	if(net === undefined){
		console.log("Loading PoseNet!");
		net = await posenet.load(networkSettings);
		console.log("PoseNet Loaded!");
	}

	// Convert and Resize the input image
	let imageDataURL = req.body.url;
	let sharpImage = await sharp(dataUriToBuffer(imageDataURL));
	let metadata = await sharpImage.metadata();
	let xScale = metadata.width/imageDim;
	let yScale = metadata.height/imageDim;
	let imageBuffer = await sharpImage.resize({
		width: imageDim,
		height: imageDim,
		fit: sharp.fit.fill,
		kernel: sharp.kernel.nearest
	}).toBuffer();
	let image = new Image();
	image.src = imageBuffer;
	ctx.drawImage(image, 0, 0, imageDim, imageDim);

	// Get Pose Estimate from Network
	let pose = await net.estimateSinglePose(canvas, {
	  flipHorizontal: false
	});

	// Scale the estimate to match the original input size
	for(keypoint of pose.keypoints){
		keypoint.position.x *= xScale;
		keypoint.position.y *= yScale;
	}
	
	res.send(pose);

});

service.start(3000).then((server) => {console.log("Server Started!")})
