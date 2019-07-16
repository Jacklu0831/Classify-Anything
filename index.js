// declare mobilenet, grab webcam from html and create KNN classifier
let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function setupWebcam() {
  	return new Promise((resolve, reject) => {
    	const navigatorAny = navigator;
    	navigator.getUserMedia = navigator.getUserMedia ||
        	navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        	navigatorAny.msGetUserMedia;
    	if (navigator.getUserMedia) {
      		navigator.getUserMedia({video: true},
	        stream => {
	          	webcamElement.srcObject = stream;
	          	webcamElement.addEventListener('loadeddata',  () => resolve(), false);
	        },
        	error => reject());
	    } else {reject();}
  	});
}

async function app() {
	// load model
  	console.log('Loading mobilenet...');
  	net = await mobilenet.load();
  	console.log('Model loaded');
  	// classify
  	await setupWebcam();
  	// reads webcam image and find class index
  	const addExample = classId => {
  		// get layers in mobilenet and pass to KNN classifier
  		const activation = net.infer(webcamElement, 'conv_preds');
  		classifier.addExample(activation, classId);
  	}
  	// add example to class when clicked
  	document.getElementById('class-1').addEventListener('click', ()=>addExample(0))
  	document.getElementById('class-2').addEventListener('click', ()=>addExample(1))
  	document.getElementById('class-3').addEventListener('click', ()=>addExample(2))
    document.getElementById('class-4').addEventListener('click', ()=>addExample(3))
    document.getElementById('class-5').addEventListener('click', ()=>addExample(4))

  	while (true) {
  		if (classifier.getNumClasses() > 0){
  			const activation = net.infer(webcamElement, 'conv_preds');
  			const result = await classifier.predictClass(activation);
        // display prediction and probability
  			const classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'];
	      	document.getElementById('console').innerText =
	        	(`PREDICTION - ${classes[result.classIndex]}\n 
              PROBABILITY - ${Math.round(result.confidences[result.classIndex]*100)/100}`);
	    }
      // give it some time to breath in the infinite loop
	    await tf.nextFrame();
  	}
}

app();