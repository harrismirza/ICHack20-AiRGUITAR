const tfjs = require('@tensorflow/tfjs-node-gpu');
tfjs.enableProdMode();
const posenet = require('@tensorflow-models/posenet');

const service = require('restana')();
const bodyParser = require('body-parser');
const cors = require('cors')
const { createCanvas, loadImage, Image } = require('canvas');

const dataUriToBuffer = require('data-uri-to-buffer');

const sharp = require('sharp');

const imageDim = 250;

const networkSettings = {
	architecture: 'ResNet50',
	outputStride: 32,
	inputResolution: { width: imageDim, height: imageDim },
	quantBytes: 4,
}

var net = undefined;

const canvas = createCanvas(imageDim, imageDim);
const ctx = canvas.getContext('2d');


service.use(cors());
service.use(bodyParser.json({limit: '50mb'}));

service.post('/', async (req, res) => {
	if(net === undefined){
		console.log("Loading PoseNet!");
		net = await posenet.load(networkSettings);
		console.log("PoseNet Loaded!");
	}

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
	/*let blob = new Blob(imageBuffer, {type: 'image/png'});
	let imageBitmap = await createImageBitmap(blob);*/
	let image = new Image();
	image.src = imageBuffer;

	ctx.drawImage(image, 0, 0, imageDim, imageDim);

	let pose = await net.estimateSinglePose(canvas, {
	  flipHorizontal: false
	});


	for(keypoint of pose.keypoints){
		keypoint.position.x *= xScale;
		keypoint.position.y *= yScale;
	}

	res.send(pose);

});

service.start(3000).then((server) => {console.log("Server Started!")})
