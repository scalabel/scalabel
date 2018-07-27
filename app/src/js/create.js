$(document).ready(function() {
  let dashboard = $('#go_to_dashboard');
  let vendor = $('#go_to_vendor_dashboard');
  dashboard.hide();
  vendor.hide();
  $('#project-form').submit(function(e) {
    e.preventDefault();

    let x = new XMLHttpRequest();
    x.onreadystatechange = function() {
      if (x.readyState === 4) {
        if (x.response) {
          alert(x.response);
        } else {
          let projectName = document.getElementById('project_name');
          projectName.value = projectName.value.replace(
            new RegExp(' ', 'g'), '_');
          dashboard[0].href = './dashboard?project_name=' + projectName.value;
          vendor[0].href = './vendor?project_name=' + projectName.value;
          dashboard.show();
          vendor.show();
        }
      }
    };
    x.open('POST', './postProject');
    x.send(new FormData(this));
  });

  let itemSelect = document.getElementById('item_type');
  let labelSelect = document.getElementById('label_type');
  let frameRate = document.getElementById('frame_rate');
  let taskSize = document.getElementById(('task_size'));
  let interpolationModeDiv = document.getElementById('interpolation_mode_div');
  let advancedOptionsButton = document.getElementById('show_advanced_options');
  let advancedOptionsDiv = document.getElementById('advanced_options');
  let demoModeCheckbox = document.getElementById('demo_mode');
  // disable all label options until item is picked
  for (let i = 1; i < labelSelect.options.length; i++) {
    labelSelect.options[i].disabled = true;
  }
  itemSelect.onchange = function() {
    // disable all label options
    for (let i = 1; i < labelSelect.options.length; i++) {
      labelSelect.options[i].disabled = true;
    }
    labelSelect.selectedIndex = 0;
    // enable just the labels that are valid
    if (itemSelect.value === 'image') {
      enableOption(labelSelect, 'box2d');
      enableOption(labelSelect, 'segmentation');
      enableOption(labelSelect, 'lane');
    } else if (itemSelect.value === 'video') {
      enableOption(labelSelect, 'box2d');
      enableOption(labelSelect, 'segmentation');
    } else if (itemSelect.value === 'pointcloud') {
      enableOption(labelSelect, 'box3d');
    } else if (itemSelect.value === 'pointcloudtracking') {
        enableOption(labelSelect, 'box3d');
    }
    // add or remove the frame rate box
    if (itemSelect.value === 'video') {
      frameRate.required = true;
      frameRate.parentNode.style.display = '';
      taskSize.required = false;
      taskSize.parentNode.style.display = 'none';
      interpolationModeDiv.style.display = 'table-cell';
    } else {
      frameRate.required = false;
      frameRate.parentNode.style.display = 'none';
      taskSize.required = true;
      taskSize.parentNode.style.display = '';
      interpolationModeDiv.style.display = 'none';
    }
  };

  labelSelect.onchange = function() {
    let labelName;
    if (labelSelect.value === 'box2d') {
      labelName = '2D Bounding Box';
    } else if (labelSelect.value === 'segmentation') {
      labelName = '2D Segmentation';
    } else if (labelSelect.value === 'lane') {
      labelName = '2D Lane';
    } else if (labelSelect.value === 'box3d') {
      labelName = '3D Bounding Box';
    }
    let pageTitle = document.getElementById('page_title');
    pageTitle.value = labelName + ' Labeling Tool';
  };

  let showAdvanced = false;
  advancedOptionsDiv.style.display = 'none';
  advancedOptionsButton.onclick = function(e) {
    e.preventDefault();
    showAdvanced = !showAdvanced;
    advancedOptionsButton.innerHTML = showAdvanced ?
      'Hide Advanced Options' : 'Show Advanced Options';
    advancedOptionsDiv.style.display = showAdvanced ? '' : 'none';
  };

  demoModeCheckbox.type = 'checkbox';
  demoModeCheckbox.setAttribute('data-on-color', 'info');
  demoModeCheckbox.setAttribute('data-on-text', 'Yes');
  demoModeCheckbox.setAttribute('data-off-text', 'No');
  demoModeCheckbox.setAttribute('data-size', 'small');
  demoModeCheckbox.setAttribute('data-label-text', 'Demo Mode');
  $('#demo_mode').bootstrapSwitch();

  /**
   * Enable the specified option in the specified select.
   * @param {object} select - The html select.
   * @param {string} optionName - The name of the option to enable.
   */
  function enableOption(select, optionName) {
    for (let i = 0; i < select.options.length; i++) {
      if (select.options[i].value === optionName) {
        select.options[i].disabled = false;
      }
    }
  }
});
