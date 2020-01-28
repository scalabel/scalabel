import { BLUE, GREEN, RED } from '../common'
import Label3D from '../label3d'
import { Controller } from './controller'
import { ScaleAxis } from './scale_axis'

/**
 * perform scaling ops
 */
export class ScaleControl extends Controller {
  constructor (labels: Label3D[], bounds: THREE.Box3) {
    super(labels, bounds)
    this._controlUnits.push(
      new ScaleAxis(
        'x',
        false,
        RED
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'y',
        false,
        GREEN
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'z',
        false,
        BLUE
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'x',
        true,
        RED
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'y',
        true,
        GREEN
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'z',
        true,
        BLUE
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
    this._local = true
  }
}
