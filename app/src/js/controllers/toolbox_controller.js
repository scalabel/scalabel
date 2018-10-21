import $ from 'jquery';
// $FlowFixMe
import 'bootstrap-switch';
import {tagImage} from '../actions/action_creators';

/**
 * Initialize Toolbox
 * @param {Object} store
 */
export function Toolbox(store: Object) {
  let self = this;
  self.store = store;
  let state = store.getState().present;
  // let categories = state.config.categories;
  let attributes = state.config.attributes;
  // FIXME: initialize all categories
  // appendCascadeCategories(categories, 0);
  // initialize all the attribute selectors
  for (let i = 0; i < attributes.length; i++) {
    let attributeInput: any = document.getElementById('custom_attribute_' +
      attributes[i].name);
    if (attributes[i].toolType === 'switch') {
      attributeInput.type = 'checkbox';
      attributeInput.setAttribute('data-on-color', 'info');
      attributeInput.setAttribute('data-on-text', 'Yes');
      attributeInput.setAttribute('data-off-text', 'No');
      attributeInput.setAttribute('data-size', 'small');
      attributeInput.setAttribute('data-label-text', attributes[i].name);
      // $FlowFixMe
      $('#custom_attribute_' + attributes[i].name).bootstrapSwitch();
    } else if (attributes[i].toolType === 'list') {
      let listOuterHtml = '<span>' + attributes[i].name + '</span>';
      listOuterHtml += '<div id="radios" class="btn-group ';
      listOuterHtml += 'attribute-btns" data-toggle="buttons">';
      for (let j = 0; j < attributes[i].values.length; j++) {
        listOuterHtml +=
          '<button id="custom_attributeselector_' + i + '-' + j +
          '" class="btn btn-raised btn-attribute btn-' +
          attributes[i].buttonColors[j] +
          '"> <input type="radio"/>' + attributes[i].values[j] +
          '</button>';
      }
      attributeInput.outerHTML = listOuterHtml;
    } else {
      attributeInput.innerHTML = 'Error: invalid tool type "' +
        attributes[i].toolType + '"';
    }
  }
  // FIXME: hook up the save listener else where
  // document.getElementById('save_btn').onclick = function() {
  //   save();
  // };
  for (let i = 0; i < attributes.length; i++) {
    for (let j = 0; j < attributes[i].values.length; j++) {
      $('#custom_attributeselector_' + i + '-' + j).on('click',
        function(e) {
          e.preventDefault();
          let currItem = self.store.getState().present.current.item;
          self.store.dispatch(tagImage(currItem, attributes[i].name, [j]));
        });
    }
  }
}
