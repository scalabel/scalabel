/* global itemType */

import {Sat3dTracker} from './point_cloud_tracking';
import {Sat3d} from './sat3d';

$(document).ready(function() {
  if (itemType == 'pointcloudtracking') {
    document.getElementById('end_btn').style.visibility = 'visible';
    new Sat3dTracker();
  } else {
    new Sat3d();
  }
});
$('body').show();
