import {tagImage} from '../action/creators';
import {BaseController} from './base_controller';

/**
 * Toolbox controller
 */
export class ToolboxController extends BaseController {
  /**
   * Select attribute
   * @param {number} attributeIndex
   * @param {Array<number>} selectedIndex
   */
   selectAttribute(
      attributeIndex: number, selectedIndex: Array<number>): void {
    let currItem = self.store.getState().present.current.item;
    // TODO change attributeName to attributeIndex
    this.dispatch(tagImage(
        currItem, attributeIndex,
        selectedIndex));
  }
}
