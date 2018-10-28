/**
 * Waits to makes sure data is received then displays timing data
 * @param {array} timingData - the data collected
 * @param {int} numMessages - the number of messages expected
 */
function processDataAfterCompletion(
    timingData: Array<any>, numMessages: number) {
  if (numMessages !== timingData.length) {
    window.setTimeout(processDataAfterCompletion,
        5, timingData, numMessages);
  } else {
    let roundTripTimeSum = 0;
    let webSocketTimeSum = 0;
    let grpcTimeSum = 0;
    let backEndTimeSum = 0;
    for (let i = 0; i < timingData.length; i++) {
      let data = timingData[i];
      let finalTime = data['finalTime'];
      let startTime = parseFloat(data['startTime']);
      let grpcDuration = parseFloat(data['grpcDuration']);
      let backEndDuration = parseFloat(data['modelServerDuration']);

      let roundTripTime = finalTime - startTime;
      roundTripTimeSum += roundTripTime;
      webSocketTimeSum += (roundTripTime - grpcDuration);
      grpcTimeSum += (grpcDuration - backEndDuration);
      backEndTimeSum += backEndDuration;
    }
    let roundTripTimeAvg = (roundTripTimeSum / numMessages).toFixed(3);
    let webSocketTimeAvg = (webSocketTimeSum / numMessages).toFixed(3);
    let grpcTimeAvg = (grpcTimeSum / numMessages).toFixed(3);
    let backEndTimeAvg = (backEndTimeSum / numMessages).toFixed(3);
    let message = timingData[0]['echoedMessage'];
    let timestamp = timingData[0]['modelServerTimestamp'];

    window.alert(
        `Message: ${message}
            Server time: ${timestamp}
            Durations (in milliseconds) over ${numMessages} trials:
                Roundtrip total: ${roundTripTimeAvg}
                Websocket server time: ${webSocketTimeAvg}
                Grpc call: ${grpcTimeAvg}
                Model server backend: ${backEndTimeAvg}
            `);
  }
}

window.onload = function() {
    let connectButton = document.getElementById('connectButton');
    let messageButton = document.getElementById('messageButton');

    if (connectButton) {
        connectButton.onclick = function() {
            requestGateInfo('12345');
        };
    }

    if (messageButton) {
        messageButton.onclick = function() {
            sendData();
        };
    }
};

/**
 * Set up gate info
 * @param {string} sessionId - The ID of the session
 */
function requestGateInfo(sessionId: string) {
  let xhr = new XMLHttpRequest();
  xhr.open('GET', './gateway');
  xhr.onreadystatechange = function() {
    if (this.readyState === 4 && this.status === 200) {
      let data = JSON.parse(this.responseText);
      let addr = data['Addr'];
      let port = data['Port'];
      registerWebsocket(sessionId, addr, port);
    }
  };
  xhr.send();
}

/**
 * Registers the session with a websocket server
 * @param {string} sessionId - The ID of the session
 * @param {string} addr - Address of the gateway server
 * @param {string} port - Port of the gateway server
 */
function registerWebsocket(sessionId: string, addr: string, port: string) {
  window.websocket = new WebSocket(`ws://${addr}:${port}/register`);
  window.websocket.onopen = function() {
    window.websocket.send(JSON.stringify({
      sessionId: sessionId,
      startTime: window.performance.now().toString(),
    }));
  };
  let numMessages = 500;
  let timingData = [];
  let currentNum = numMessages;
  window.websocket.onmessage = function(e) {
    let finalTime = window.performance.now();
    let data = {};
    if (typeof e.data === 'string') {
      data = JSON.parse(e.data);
    }
    if (data['sessionId']) {
      data['timingData']['finalTime'] = finalTime;
      window.alert(`Registered under id ${data['sessionId']}`);
      let registrationTimingData = [];
      registrationTimingData.push(data['timingData']);
      processDataAfterCompletion(registrationTimingData, 1);
    }
    if (data['echoedMessage']) {
      data['finalTime'] = finalTime;
      timingData.push(data);
      if (currentNum > 1) {
        currentNum -= 1;
        sendData();
      } else {
        processDataAfterCompletion(timingData, numMessages);
        currentNum = numMessages;
        timingData = [];
      }
    }
  };
  window.websocket.onclose = function() {
  };
}

/**
 * Send a message through the web socket
 */
function sendData() {
  window.websocket.send(JSON.stringify({
    message: $('#message').val(),
    startTime: window.performance.now().toString(),
  }));
}
