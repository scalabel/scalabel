/* flow */

import {sprintf} from 'sprintf-js';

$(document).ready(function() {
  let tds = document.getElementById('main_table').getElementsByTagName('td');
  let totalTaskLabeled = 0;
  let totalLabels = 0;
  for (let i = 0; i < tds.length; i++) {
    if (tds[i].className === 'countLabeledImage') {
      totalTaskLabeled += parseInt(tds[i].innerHTML) > 0 ? 1 : 0;
    }
    if (tds[i].className === 'countLabelInTask') {
      totalLabels += parseInt(tds[i].innerHTML);
    }
  }
  document.getElementById('totalTaskLabeled').textContent =
      sprintf('%d', totalTaskLabeled);
  document.getElementById('totalLabels').textContent =
      sprintf('%d', totalLabels);

  $('body').show();
});
