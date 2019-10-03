import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import Session from '../../common/session'
import { Attribute } from '../../functional/types'
import { Context2D } from '../util'
import { Label2D } from './label2d'

/**
 * Tag2D drawable class
 */
export class Tag2D extends Label2D {
  /** attributes for task */
  public configAttributes: Attribute[]

  constructor () {
    super()
    this.configAttributes = Session.getState().task.config.attributes
  }

  /**
   * no-op
   */
  public commitLabel () {
    return false
  }

  /**
   * no-op
   */
  public onMouseMove () {
    return false
  }

  /**
   * no-op
   */
  public onKeyDown (): boolean {
    return true
  }

  /**
   * no-op
   */
  public onKeyUp (): void {
    return
  }

  /**
   * no-op
   */
  public updateShapes () {
    return
  }

  /**
   * Draws tag box
   * @param context
   */
  public draw (context: Context2D) {
    context.font = '36px Arial'
    const abbr: string[] = []
    for (const key in this.attributes) {
      if (this.attributes[key][0] !== -1) {
        const selectedIndex = this.attributes[key][0]
        const selectedAttribute = this.configAttributes[key]
        abbr.push(sprintf('  %s: %s', selectedAttribute.name,
          selectedAttribute.values[selectedIndex]))
      }
    }
    context.fillStyle = 'lightgrey'
    context.globalAlpha = 0.3
    context.fillRect(5, 5, 400,
      (abbr.length) ? abbr.length * 35 + 15 : 0)
    context.fillStyle = 'red'
    context.globalAlpha = 1.0
    for (let i = 0; i < abbr.length; i++) {
      context.fillText(abbr[i], 5, 40 + i * 35)
    }
  }

  /**
   * no-op
   */
  public initTemp () {
    return
  }
}
