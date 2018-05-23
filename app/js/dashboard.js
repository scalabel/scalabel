/**
 * TODO
 * @param {string} filename
 * @param {string} text
 */
function download(filename, text) {
  let element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,'
    + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

/**
 * TODO
 * @param {string} id
 * @return {string}
 */
function getID(id) {
  let str = id.toString();
  let len = str.length;
  for (let i = 0; i < (4 - len); i++) {
    str = '0' + str;
  }
  return str;
}

$(document).ready(function() {
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function () {
    if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200
      && xhr.response) {
      let tasks = JSON.parse(xhr.response);
    }
  };
  xhr.open('POST', './postGetTasksHandler');

  // $('.panel-heading').text(projectName);
});
