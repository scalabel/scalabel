/* flow */

$(document).ready(function() {
  let tds = document.getElementById('main_table').getElementsByTagName('td');
  for (let i = 0; i < tds.length; i++) {
    if (tds[i].className === 'submitted') {
      if (tds[i].innerHTML === 'true') {
        tds[i].innerHTML = '<i class="fas fa-check"></i>';
      } else {
        tds[i].innerHTML = '';
      }
    }
  }
  $('body').show();
});
