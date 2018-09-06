/* global itemType */

import {Sat3dTracker} from './point_cloud_tracking';
import {Sat3d} from './sat3d';
import {Box3d} from './box3d';
import 'bootstrap-material-design';

$(document).ready(function() {
  $('body').bootstrapMaterialDesign();
  if (itemType === 'pointcloudtracking') {
    document.getElementById('end_btn').style.visibility = 'visible';
    new Sat3dTracker();
  } else {
    new Sat3d(Box3d);
  }
});
$('body').show();
