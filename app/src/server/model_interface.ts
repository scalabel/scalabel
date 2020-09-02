import { addPolygon2dLabel } from "../action/polygon2d"
import { ModelEndpoint } from "../const/connection"
import {
  makeItemExport,
  makeLabelExport,
  makeSimplePathPoint2D
} from "../functional/states"
import { AddLabelsAction } from "../types/action"
import { ModelQuery } from "../types/message"
import { PathPoint2DType, PathPointType, RectType } from "../types/state"
import { convertPolygonToExport } from "./export"

/**
 * API between redux style data and data for the models
 */
export class ModelInterface {
  /** project name */
  public projectName: string
  /** current session id */
  public sessionId: string

  /**
   * Constructor
   *
   * @param projectName
   * @param sessionId
   */
  constructor(projectName: string, sessionId: string) {
    this.projectName = projectName
    this.sessionId = sessionId
  }

  /**
   * Query for 'rect -> polygon' segmentation
   *
   * @param rect
   * @param url
   * @param itemIndex
   */
  public makeRectQuery(
    rect: RectType,
    url: string,
    itemIndex: number
  ): ModelQuery {
    const label = makeLabelExport({
      box2d: rect
    })
    const item = makeItemExport({
      name: this.projectName,
      url,
      labels: [label]
    })
    return {
      data: item,
      endpoint: ModelEndpoint.PREDICT_POLY,
      itemIndex
    }
  }

  /**
   * Query for refining 'polygon -> polygon' segmentation
   *
   * @param points
   * @param url
   * @param itemIndex
   * @param labelType
   */
  public makePolyQuery(
    points: PathPoint2DType[],
    url: string,
    itemIndex: number,
    labelType: string
  ): ModelQuery {
    const poly2d = convertPolygonToExport(points, labelType)
    const label = makeLabelExport({
      poly2d
    })
    const item = makeItemExport({
      name: this.projectName,
      url,
      labels: [label]
    })
    return {
      data: item,
      endpoint: ModelEndpoint.REFINE_POLY,
      itemIndex
    }
  }

  /**
   * Translate polygon response to an action
   *
   * @param polyPoints
   * @param itemIndex
   */
  public makePolyAction(
    polyPoints: number[][],
    itemIndex: number
  ): AddLabelsAction {
    const points = polyPoints.map((point: number[]) => {
      return makeSimplePathPoint2D(point[0], point[1], PathPointType.LINE)
    })

    const action = addPolygon2dLabel(itemIndex, -1, [0], points, true, false)
    action.sessionId = this.sessionId
    return action
  }
}
