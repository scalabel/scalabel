import {sprintf} from 'sprintf-js';
import Session from '../common/session.js';
import {BaseViewer} from './base_viewer';
import {TitleBarController} from '../controllers/title_bar_controller';

/**
 * TitleBarViewer class
 * Currently only responsible for counting labels
 */
export class TitleBarViewer extends BaseViewer {
  /**
   * @param {TitleBarController} controller
   * @constructor
   */
  constructor(controller: TitleBarController) {
    super(controller);
    if (Session.getState().config.labelType === 'tag') {
      let labelCountTitle = document.getElementById('label-count-title');
      if (labelCountTitle) {
        labelCountTitle.style.visibility = 'hidden';
      }
      let labelCount = document.getElementById('label-count');
      if (labelCount) {
        labelCount.style.visibility = 'hidden';
      }
    }
    // buttons
    let prevBtn = document.getElementById('prev_btn');
    if (prevBtn) {
      prevBtn.onclick = controller.goToPreviousItem.bind(controller);
    }
    let nextBtn = document.getElementById('next_btn');
    if (nextBtn) {
      nextBtn.onclick = controller.goToNextItem.bind(controller);
    }
    let saveBtn = document.getElementById('save-btn');
    if (saveBtn) {
      saveBtn.onclick = controller.save.bind(controller);
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
    let currItem = state.current.item;
    let numLabels = state.items[currItem].labels.length;
    let labelCount = document.getElementById('label-count');
    if (labelCount) {
      labelCount.textContent = sprintf('%d', numLabels);
    }
    return true;
  }
}
