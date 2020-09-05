import { AttributeToolType } from "../../const/common"
import {
  Attribute,
  IdType,
  LabelType,
  ShapeType,
  State
} from "../../types/state"
import { Context2D } from "../util"
import { DrawMode, Label2D } from "./label2d"
import { Label2DList } from "./label2d_list"

/**
 * Tag2D drawable class
 */
export class Tag2D extends Label2D {
  /** attributes for task */
  public configAttributes: Attribute[]

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label2DList) {
    super(labelList)
    this.configAttributes = []
  }

  /** Get shape objects for committing to state */
  public shapes(): ShapeType[] {
    return []
  }

  /**
   * no-op
   */
  public onMouseMove(): boolean {
    return false
  }

  /**
   * no-op
   */
  public onKeyDown(): boolean {
    return true
  }

  /**
   * no-op
   */
  public onKeyUp(): void {}

  /**
   * no-op
   */
  public updateShapes(): void {}

  /**
   * Convert label state to drawable
   *
   * @param state
   * @param itemIndex
   * @param labelId
   */
  public updateState(state: State, itemIndex: number, labelId: IdType): void {
    super.updateState(state, itemIndex, labelId)
    this.configAttributes = state.task.config.attributes
  }

  /**
   * Draws tag box
   *
   * @param context
   * @param _ratio
   * @param mode
   */
  public draw(context: Context2D, _ratio: number, mode: DrawMode): void {
    if (mode === DrawMode.VIEW) {
      context.font = "36px Arial"
      const abbr: string[] = []
      for (const key in this.attributes) {
        if (this.attributes[key][0] !== -1) {
          const selectedIndex = this.attributes[key][0]
          const selectedAttribute = this.configAttributes[key]
          if (selectedAttribute.toolType === AttributeToolType.SWITCH) {
            if (selectedIndex === 1) {
              abbr.push(
                ` ${selectedAttribute.name}: ${selectedAttribute.tagText}`
              )
            }
          } else {
            abbr.push(
              `  ${selectedAttribute.name}: ${selectedAttribute.values[selectedIndex]}`
            )
          }
        }
      }
      context.fillStyle = "lightgrey"
      context.globalAlpha = 0.3
      context.fillRect(5, 5, 400, abbr.length > 0 ? abbr.length * 35 + 15 : 0)
      context.fillStyle = "red"
      context.globalAlpha = 1.0
      for (let i = 0; i < abbr.length; i++) {
        context.fillText(abbr[i], 5, 40 + i * 35)
      }
    }
  }

  /**
   * no op
   *
   * @param state
   * @param _state
   * @param _start
   */
  protected initTempLabel(): LabelType {
    throw new Error("Method not implemented.")
  }
}
