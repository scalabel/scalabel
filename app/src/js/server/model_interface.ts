import { addPolygon2dLabel } from '../action/polygon2d'
import { AddLabelsAction } from '../action/types'
import { makeItemExport, makeLabelExport, makeSimplePathPoint2D } from '../functional/states'
import { PathPointType, PolygonType, RectType } from '../functional/types'
import { convertPolygonToExport } from './export'
import { ModelEndpoint, ModelQuery } from './types'

/**
 * API between redux style data and data for the models
 */
export class ModelInterface {
  /** project name */
  public projectName: string
  /** current session id */
  public sessionId: string

  constructor (projectName: string, sessionId: string) {
    this.projectName = projectName
    this.sessionId = sessionId
  }

  /**
   * Query for 'rect -> polygon' segmentation
   */
  public makeRectQuery (
    rect: RectType, url: string, itemIndex: number): ModelQuery {
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
   */
  public makePolyQuery (
    poly: PolygonType, url: string,
    itemIndex: number, labelType: string): ModelQuery {
    const poly2d = convertPolygonToExport(poly, labelType)
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
   */
  public makePolyAction (
    polyPoints: number[][], itemIndex: number): AddLabelsAction {
    const points = polyPoints.map((point: number[]) => {
      return makeSimplePathPoint2D(
          point[0], point[1], PathPointType.LINE)
    })

    const action = addPolygon2dLabel(
      itemIndex, -1, [0], points, true, false
    )
    action.sessionId = this.sessionId
    return action
  }
}
