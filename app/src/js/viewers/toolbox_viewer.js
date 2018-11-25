import $ from 'jquery';
import Session from '../common/session_single';
import {BaseViewer} from './base_viewer';
/* :: import {ToolboxController} from '../controllers/toolbox_controller'; */
import {sprintf} from 'sprintf-js';
import 'bootstrap-switch';

/**
 * ToolboxViewer class
 * responsible for tool box
 */
export class ToolboxViewer extends BaseViewer {
  /**
   * @param {ToolboxController} controller: reference to controller
   * @constructor
   */
  constructor(controller/* : ToolboxController */) {
    super(controller);
    let self = this;
    let categories: Array<string> = Session.getState().config.categories;
    if (categories === null || categories.length === 0) {
      let customCategories = document.getElementById('custom_categories');
      if (customCategories) {
        customCategories.style.visibility = 'hidden';
      }
    }
    let attributes = Session.getState().config.attributes;
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
        let listOuterHtml = sprintf('<span>%s</span>', attributes[i].name);
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
    // document.getElementById('save-btn').onclick = function() {
    //   save();
    // };
    for (let i = 0; i < attributes.length; i++) {
      for (let j = 0; j < attributes[i].values.length; j++) {
        $('#custom_attributeselector_' + i + '-' + j).on('click',
            function(e) {
              e.preventDefault();
              // TODO: [j] should have multiple elements for multi-level
              // selection
              self.controller.selectAttribute(i, [j]);
            });
      }
    }
  }

  /**
   * Redraw the image canvas.
   * @return {boolean}
   */
  redraw(): boolean {
    if (!super.redraw()) {
      return false;
    }
    let state = this.state;
    let currentItemIndex = state.current.item;
    let currentItem = state.items[currentItemIndex];
    let label = state.labels[currentItem.labels[0]];
    let attributesMap = state.config.attributes;
    for (let i = 0; i < attributesMap.length; i++) {
      for (let j = 0; j < attributesMap[i].values.length; j++) {
        let selector = $('#custom_attributeselector_' + i + '-' + j);
        let selectedIndices = null;
        if (label && label.attributes) {
          selectedIndices = label.attributes[attributesMap[i].name];
        }
        if (label && label.attributes
          && selectedIndices && selectedIndices[0] === j) {
          selector.addClass('active');
        } else {
          selector.removeClass('active');
        }
      }
    }
    return true;
  }
}
