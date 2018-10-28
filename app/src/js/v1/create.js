import 'bootstrap-switch';

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
    // eslint-disable-next-line
    x.send(new FormData(this));
  });

  let itemSelect = document.getElementById('item_type');
  let labelSelect = document.getElementById('label_type');
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
      enableOption(labelSelect, 'tag');
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
      taskSize.required = false;
      taskSize.parentNode.style.display = 'none';
      interpolationModeDiv.style.display = 'table-cell';
    } else {
      taskSize.required = true;
      taskSize.parentNode.style.display = '';
      interpolationModeDiv.style.display = 'none';
    }
  };

  labelSelect.onchange = function() {
    let labelName;
    let instructions;
    if (labelSelect.value === 'tag') {
      labelName = 'Image Tagging';
      document.getElementById('categories').style.visibility = 'hidden';
      document.getElementById('categories_label').style.visibility = 'hidden';
    } else if (labelSelect.value === 'box2d') {
      labelName = '2D Bounding Box';
      instructions = 'http://data-bdd.berkeley.edu/label/bbox/instruction.html';
      document.getElementById('categories').style.visibility = 'visible';
      document.getElementById('categories_label').style.visibility = 'visible';
    } else if (labelSelect.value === 'segmentation') {
      labelName = '2D Segmentation';
      instructions = 'http://data-bdd.berkeley.edu/label/seg/readme.html';
      document.getElementById('categories').style.visibility = 'visible';
      document.getElementById('categories_label').style.visibility = 'visible';
    } else if (labelSelect.value === 'lane') {
      labelName = '2D Lane';
      instructions = 'http://data-bdd.berkeley.edu/label/seg/readme.html';
      document.getElementById('categories').style.visibility = 'visible';
      document.getElementById('categories_label').style.visibility = 'visible';
    } else if (labelSelect.value === 'box3d') {
      labelName = '3D Bounding Box';
      document.getElementById('categories').style.visibility = 'visible';
      document.getElementById('categories_label').style.visibility = 'visible';
    }
    document.getElementById('page_title').value = labelName + ' Labeling Tool';
    document.getElementById('instructions').value = instructions;
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
  demoModeCheckbox.setAttribute('data-size', 'medium');
  demoModeCheckbox.setAttribute('data-label-text', 'Demo');
  $('#demo_mode').bootstrapSwitch('state', false);

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
