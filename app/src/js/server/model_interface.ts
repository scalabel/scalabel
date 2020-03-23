import { addPolygon2dLabel } from '../action/polygon2d'
import { AddLabelsAction } from '../action/types'
import { PathPoint2D, PointType } from '../drawable/2d/path_point2d'
import { makeItemExport, makeLabelExport } from '../functional/states'
import { PolygonType, RectType } from '../functional/types'
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
   * Query for polygon rnn 'rect -> polygon' prediction
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
      endpoint: ModelEndpoint.POLYGON_RNN_BASE,
      itemIndex
    }
  }

  /**
   * Query for polygon rnn 'polygon -> polygon' prediction
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
      endpoint: ModelEndpoint.POLYGON_RNN_REFINE,
      itemIndex
    }
  }

  /**
   * Translate polygon response to an action
   */
  public makePolyAction (
    polyPoints: number[][], itemIndex: number): AddLabelsAction {
    const points = polyPoints.map((point: number[]) => {
      return (new PathPoint2D(
          point[0], point[1], PointType.VERTEX)).toPathPoint()
    })

    const action = addPolygon2dLabel(
      itemIndex, -1, [0], points, true, false
    )
    action.sessionId = this.sessionId
    return action
  }
}
