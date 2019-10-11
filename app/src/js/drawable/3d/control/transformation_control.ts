import * as THREE from 'three'
import { Key } from '../../../common/types'
import { projectionFromNDC } from '../../../view_config/point_cloud'
import { Controller } from './controller'
import { RotationControl } from './rotation_control'
import { ScaleControl } from './scale_control'
import { TranslationControl } from './translation_control'

/**
 * Group TranslationControl, RotationControl, and ScaleControl together
 */
export class TransformationControl extends THREE.Group {
  /** Current controller */
  private _currentController: Controller
  /** Translation controller */
  private _translationControl: TranslationControl
  /** Rotation controller */
  private _rotationControl: RotationControl
  /** Scale controller */
  private _scaleControl: ScaleControl
  /** Attached object */
  private _object: THREE.Object3D | null

  constructor () {
    super()
    this._translationControl = new TranslationControl()
    this._rotationControl = new RotationControl()
    this._scaleControl = new ScaleControl()
    this._currentController = this._rotationControl
    this.add(this._currentController)
    this._object = null
  }

  /**
   * Highlight correct axis
   * @param intersection
   */
  public setHighlighted (intersection?: THREE.Intersection) {
    this._currentController.setHighlighted(intersection)
  }

  /**
   * Mouse down
   */
  public onMouseDown (camera: THREE.Camera) {
    return this._currentController.onMouseDown(camera)
  }

  /**
   * Handle key events
   * @param e
   */
  public onKeyDown (e: KeyboardEvent): boolean {
    switch (e.key) {
      case Key.Q_UP:
      case Key.Q_LOW:
        this._currentController.toggleFrame()
        return true
      case Key.T_UP:
      case Key.T_LOW:
        this.switchController(this._translationControl)
        return true
      case Key.R_UP:
      case Key.R_LOW:
        this.switchController(this._rotationControl)
        return true
      case Key.S_UP:
      case Key.S_LOW:
        this.switchController(this._scaleControl)
        return true
      case Key.F_UP:
      case Key.F_LOW:
        return true
    }
    return false
  }

  /**
   * Mouse movement while mouse down on box (from raycast)
   * @param x: NDC
   * @param y: NDC
   */
  public onMouseMove (x: number, y: number, camera: THREE.Camera) {
    const projection = projectionFromNDC(x, y, camera)
    return this._currentController.onMouseMove(projection)
  }

  /**
   * Mouse up
   */
  public onMouseUp () {
    return this._currentController.onMouseUp()
  }

  /**
   * Override ThreeJS raycast to intersect with box
   * @param raycaster
   * @param intersects
   */
  public raycast (
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ) {
    this._currentController.raycast(raycaster, intersects)
  }

  /**
   * Attach to object
   * @param object
   */
  public attach (object: THREE.Object3D) {
    this.updateMatrix()
    this.updateMatrixWorld(true)
    this._currentController.attach(object)
    this._object = object
  }

  /**
   * Detach
   */
  public detach () {
    this._currentController.detach()
    this._object = null
  }

  /**
   * Whether currently attached
   */
  public attached () {
    return this._object !== null
  }

  /**
   * Switch to new controller
   * @param controller
   */
  private switchController (controller: Controller) {
    if (!this._object) {
      return
    }
    const object = this._object
    this.remove(this._currentController)
    this.detach()

    this._currentController = controller
    this.add(this._currentController)
    this.attach(object)
  }
}
