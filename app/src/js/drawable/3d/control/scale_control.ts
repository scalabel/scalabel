
import * as THREE from 'three'
import { Controller } from './controller'
import { ScaleAxis } from './scale_axis'

/**
 * Groups TranslationAxis's and TranslationPlanes to perform translation ops
 */
export class ScaleControl extends Controller {
  constructor (camera: THREE.Camera) {
    super(camera)
    this._controlUnits.push(
      new ScaleAxis(
        'x',
        false,
        0xff0000
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'y',
        false,
        0x00ff00
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'z',
        false,
        0x0000ff
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'x',
        true,
        0xff0000
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'y',
        true,
        0x00ff00
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'z',
        true,
        0x0000ff
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
    this._local = true
  }

  /** Toggle local/world */
  public toggleFrame () {
    this._local = true
  }
}
