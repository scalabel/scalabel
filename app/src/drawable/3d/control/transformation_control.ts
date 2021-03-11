import * as THREE from "three"

import { Key } from "../../../const/common"
import { projectionFromNDC } from "../../../view_config/point_cloud"
import Label3D from "../label3d"
import { Controller } from "./controller"
import { RotationControl } from "./rotation_control"
import { ScaleControl } from "./scale_control"
import { TranslationControl } from "./translation_control"

/**
 * Group TranslationControl, RotationControl, and ScaleControl together
 */
export class TransformationControl extends THREE.Group {
  /** Current controller */
  private _currentController: Controller
  /** Translation controller */
  private readonly _translationControl: TranslationControl
  /** Rotation controller */
  private readonly _rotationControl: RotationControl
  /** Scale controller */
  private readonly _scaleControl: ScaleControl
  /** Labels to transform */
  private _labels: Label3D[]
  /** Bounds of the labels */
  private readonly _bounds: THREE.Box3

  /**
   * Constructor
   */
  constructor() {
    super()
    this._labels = []
    this._bounds = new THREE.Box3()
    this._translationControl = new TranslationControl(
      this._labels,
      this._bounds
    )
    this._rotationControl = new RotationControl(this._labels, this._bounds)
    this._scaleControl = new ScaleControl(this._labels, this._bounds)
    this._currentController = this._rotationControl
    this.add(this._currentController)
  }

  /**
   * Add new label to control for transforming
   *
   * @param newLabel
   */
  public addLabel(newLabel: Label3D): void {
    this._labels.push(newLabel)
    this.updateBounds()
  }

  /** Clear label array */
  public clearLabels(): void {
    this._labels.length = 0
  }

  /**
   * Highlight correct axis
   *
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): void {
    this._currentController.setHighlighted(intersection)
  }

  /**
   * Mouse down
   *
   * @param camera
   */
  public onMouseDown(camera: THREE.Camera): boolean {
    return this._currentController.onMouseDown(camera)
  }

  /**
   * Handle key events
   *
   * @param e
   * @param camera
   */
  public onKeyDown(e: KeyboardEvent, camera: THREE.Camera): boolean {
    switch (e.key) {
      case Key.Q_UP:
      case Key.Q_LOW:
        this._currentController.toggleFrame()
        this.updateBounds()
        return true
      case Key.T_UP:
      case Key.T_LOW:
        this.switchController(this._translationControl)
        return true
      case Key.R_UP:
      case Key.R_LOW:
        this.switchController(this._rotationControl)
        return true
      case Key.E_UP:
      case Key.E_LOW:
        this.switchController(this._scaleControl)
        return true
      case Key.I_UP:
      case Key.I_LOW:
        this._currentController.keyDown(e.key, camera)
        this.updateBounds()
        return true
      case Key.K_UP:
      case Key.K_LOW:
        this._currentController.keyDown(e.key, camera)
        this.updateBounds()
        return true
      case Key.J_UP:
      case Key.J_LOW:
        this._currentController.keyDown(e.key, camera)
        this.updateBounds()
        return true
      case Key.L_UP:
      case Key.L_LOW:
        this._currentController.keyDown(e.key, camera)
        this.updateBounds()
        return true
      case Key.SHIFT:
        this._currentController.keyDown(e.key, camera)
    }
    return false
  }

  /**
   * Handle key up
   *
   * @param e
   */
  public onKeyUp(e: KeyboardEvent): void {
    this._currentController.keyUp(e.key)
  }

  /**
   * Mouse movement while mouse down on box (from raycast)
   *
   * @param x: NDC
   * @param y: NDC
   * @param x
   * @param y
   * @param camera
   */
  public onMouseMove(x: number, y: number, camera: THREE.Camera): boolean {
    const projection = projectionFromNDC(x, y, camera)
    const result = this._currentController.onMouseMove(projection)
    this.updateBounds()
    return result
  }

  /**
   * Mouse up
   */
  public onMouseUp(): boolean {
    return this._currentController.onMouseUp()
  }

  /**
   * Override ThreeJS raycast to intersect with box
   *
   * @param raycaster
   * @param intersects
   */
  public raycast(
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ): void {
    this._currentController.raycast(raycaster, intersects)
  }

  /** Returns whether control is highlighted */
  public get highlighted(): boolean {
    return this._currentController.highlighted
  }

  /**
   * Switch to new controller
   *
   * @param controller
   */
  private switchController(controller: Controller): void {
    this.remove(this._currentController)
    this._currentController = controller
    this.add(this._currentController)
    this.updateScale()
  }

  /** Update bounds of the transformation control */
  private updateBounds(): void {
    this._bounds.makeEmpty()
    for (const label of this._labels) {
      this._bounds.union(label.bounds(this._currentController.local))
    }
    this._bounds.getCenter(this.position)
    if (this._currentController.local) {
      this.quaternion.copy(this._labels[0].orientation)
    } else {
      this.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0)
    }

    this.updateScale()
  }

  /** Update controller scale */
  private updateScale(): void {
    const size = new THREE.Vector3()
    this._bounds.getSize(size)
    this._currentController.updateScale(size)
  }
}
