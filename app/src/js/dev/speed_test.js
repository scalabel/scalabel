import {sprintf} from 'sprintf-js';

/**
 * @param {Object.<string, array>} data - map from category to data
 * @return {Object.<string, number>} - map from category to average
 */
function datasToAverages(
    data: { [string]: Array<any> },
) {
    let averages = {};
    Object.keys(data).forEach(function(category) {
        let sum = data[category].reduce((a, b) => a + b, 0);
        let numData = data[category].length;
        averages[category] = sum/numData;
    });
    return averages;
}

/**
 * @param {Object.<string, number>} avgTimesTotal - map from timing
 * category to data, must contain all relevant fields
 * @param {string} experimentType - string label for experiment
 */
function addRowToTable(
    avgTimesTotal: { [string]: number }, experimentType: string,
) {
    let tableElement = (document.getElementById('resultsTable'): any);
    let table = (tableElement: HTMLTableElement);

    let row = table.insertRow(window.currentRowNumber);
    let typeCell = row.insertCell(0);
    let roundtripCell = row.insertCell(1);
    let websocketCell = row.insertCell(2);
    let grpcCell = row.insertCell(3);
    let backendCell = row.insertCell(4);

    typeCell.innerHTML = experimentType;
    roundtripCell.innerHTML = avgTimesTotal.roundTrip.toFixed(3);
    websocketCell.innerHTML = avgTimesTotal.webSocket.toFixed(3);
    grpcCell.innerHTML = avgTimesTotal.grpc.toFixed(3);
    backendCell.innerHTML = avgTimesTotal.backEnd.toFixed(3);

    window.currentRowNumber += 1;
}

/**
 Initialize/reset the global timing data for all web socket sessions
 */
function clearTimingData() {
    window.allTimingData = {
        roundTrip: [],
        webSocket: [],
        grpc: [],
        backEnd: [],
    };
}

/**
 * Waits to makes sure data is received then displays timing data
 * @param {array} sessionTimingData - the data collected
 * @param {int} testType - either init or speedTest
 */
function processDataAfterCompletion(
    sessionTimingData: Array<any>, testType: string) {
  let numMessagesExpected = window.numMessages;
  if (testType === 'init') {
      numMessagesExpected = 1;
  }
  if (numMessagesExpected !== sessionTimingData.length) {
    /* wait until all the messages return
       i.e. until length equals number expected */
    window.setTimeout(processDataAfterCompletion,
        5, sessionTimingData, testType);
  } else {
    let formattedSessionTimingData = {
        roundTrip: [],
        webSocket: [],
        grpc: [],
        backEnd: [],
    };

    for (let i = 0; i < sessionTimingData.length; i++) {
      // format data for 1 trial of 1 session
      let data = sessionTimingData[i];
      let finalTime = data['finalTime'];
      let startTime = parseFloat(data['startTime']);
      let grpcDuration = parseFloat(data['grpcDuration']);
      let backEndDuration = parseFloat(data['modelServerDuration']);

      /* subtract because the durations overlap, for example:
       grpc starts, backend starts, backend ends, grpc ends */
      let roundTripTime = finalTime - startTime;
      formattedSessionTimingData.roundTrip.push(roundTripTime);
      formattedSessionTimingData.webSocket.push(roundTripTime - grpcDuration);
      formattedSessionTimingData.grpc.push(grpcDuration - backEndDuration);
      formattedSessionTimingData.backEnd.push(backEndDuration);
    }
    let sessionAvgTimes = datasToAverages(formattedSessionTimingData);

    Object.keys(sessionAvgTimes).forEach(function(category) {
        window.allTimingData[category].push(sessionAvgTimes[category]);
    });

    // wait until all sessions complete to aggregate data
    if (window.allTimingData.roundTrip.length === window.numSessions) {
        let allAvgTimes = datasToAverages(window.allTimingData);
        let numSessions = sprintf('%d sessions', window.numSessions);
        let numMessages = sprintf('%d messages', window.numMessages);
        let experimentType = sprintf('Initializing %s', numSessions);
        if (testType !== 'init') {
            experimentType = sprintf('Speed test with %s, %s per session',
                numSessions, numMessages);
        }

        addRowToTable(allAvgTimes, experimentType);

        clearTimingData();
        if (testType === 'init') {
            speedTest();
        } else {
            killAllSessions();
        }
    }
  }
}

window.onload = function() {
    let runButton = document.getElementById('runButton');

    clearTimingData();

    window.currentRowNumber = 1;

    if (runButton) {
        runButton.onclick = function() {
            requestGateInfo();
        };
    }
};

/**
 * Set up gate info
 */
function requestGateInfo() {
  let xhr = new XMLHttpRequest();
  xhr.open('GET', './gateway');
  xhr.onreadystatechange = function() {
    if (this.readyState === 4 && this.status === 200) {
      let data = JSON.parse(this.responseText);
      let addr = data['Addr'];
      let port = data['Port'];
      generateSessions(addr, port);
    }
  };
  xhr.send();
}

/**
 * Generates session IDs to register websockets
 * @param {string} addr - the websocket address
 * @param {string} port - the websocket port
 */
function generateSessions(addr: string, port: string) {
    let numSessionsInput = (document.getElementById('numSessions'): any);
    window.numSessions = parseInt((numSessionsInput: HTMLInputElement).value);

    let numMessagesInput = (document.getElementById('numMessages'): any);
    window.numMessages = parseInt((numMessagesInput: HTMLInputElement).value);

    window.websockets = [];

    let sessionIds = new Set();
    for (let i = 0; i < window.numSessions; i++) {
        let newId = String(parseInt(Math.random() * Number.MAX_SAFE_INTEGER));
        while (sessionIds.has(newId)) {
            newId = String(parseInt(Math.random() * Number.MAX_SAFE_INTEGER));
        }
        sessionIds.add(newId);
        registerWebsocket(newId, i, addr, port);
    }
}

/**
 * Registers the session with a websocket server
 * @param {string} sessionId - The ID of the session
 * @param {number} sessionIndex - The index of the session in window.websockets
 * @param {string} addr - Address of the gateway server
 * @param {string} port - Port of the gateway server
 */
function registerWebsocket(sessionId: string, sessionIndex: number,
                            addr: string, port: string) {
  let websocket = new WebSocket(`ws://${addr}:${port}/register`);
  window.websockets.push(websocket);
  websocket.onopen = function() {
    websocket.send(JSON.stringify({
      sessionId: sessionId,
      startTime: window.performance.now().toString(),
    }));
  };

  let sessionTimingData = [];
  let currentNum = window.numMessages;

  websocket.onmessage = function(e) {
    let finalTime = window.performance.now();
    let data = {};
    if (typeof e.data === 'string') {
      data = JSON.parse(e.data);
    }
    // connected
    if (data['sessionId']) {
      data['timingData']['finalTime'] = finalTime;
      let initSessionTimingData = [];
      initSessionTimingData.push(data['timingData']);
      processDataAfterCompletion(initSessionTimingData, 'init');
    }
    // sent a message
    if (data['echoedMessage']) {
      data['finalTime'] = finalTime;
      sessionTimingData.push(data);
      if (currentNum > 1) {
        currentNum -= 1;
        sendData(sessionIndex);
      } else {
        processDataAfterCompletion(sessionTimingData, 'speedTest');
        currentNum = window.numMessages;
        sessionTimingData = [];
      }
    }
  };
  websocket.onclose = function() {
  };
}

/**
 * Sends a message through each web socket
 */
function speedTest() {
    for (let i = 0; i < window.numSessions; i++) {
        sendData(i);
    }
}

/**
 * Send a message through the specified web socket
 * @param {number} sessionIndex - The index of the session in window.websockets
 */
function sendData(sessionIndex: number) {
  let messageElement = (document.getElementById('message'): any);
  let message = (messageElement: HTMLTextAreaElement).value;

  window.websockets[sessionIndex].send(JSON.stringify({
    message: sprintf('%s%d', message, sessionIndex),
    startTime: window.performance.now().toString(),
  }));
}

/**
 * Kills each web socket
 */
function killAllSessions() {
    for (let i = 0; i < window.numSessions; i++) {
        killSession(i);
    }
}

/**
 * Kills the specified web socket
 * @param {number} sessionIndex - The index of the session in window.websockets
 */
function killSession(sessionIndex: number) {
    window.websockets[sessionIndex].send(JSON.stringify({
      terminateSession: 'true',
    }));
}
