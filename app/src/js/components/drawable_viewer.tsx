import React from 'react'
import Session from '../common/session'
import { ViewerConfigType } from '../functional/types'
import { Component } from './component'

/** Generate string to use for react component key */
export function viewerReactKey (id: number) {
  return `viewer${id}`
}

interface ClassType {
  /** container */
  viewer_container: string
}

export interface ViewerProps {
  /** classes */
  classes: ClassType
  /** id of the viewer, for referencing viewer config in state */
  id: number
}

/**
 * Canvas Viewer
 */
export abstract class DrawableViewer extends Component<ViewerProps> {
  /** Moveable container */
  protected _container: HTMLDivElement | null
  /** viewer config */
  protected _viewerConfig?: ViewerConfigType
  /** viewer id */
  protected _viewerId: number

  /** UI handler */
  protected _keyDownHandler: (e: KeyboardEvent) => void
  /** UI handler */
  protected _keyUpHandler: (e: KeyboardEvent) => void
  /** UI Handler */
  protected _wheelHandler: (e: WheelEvent) => void

  /** The hashed list of keys currently down */
  protected _keyDownMap: { [key: string]: boolean }
  /** Mouse x-coord */
  protected _mX: number
  /** Mouse y-coord */
  protected _mY: number
  /** Whether mouse is down */
  protected _mouseDown: boolean
  /** which button is pressed on mouse down */
  protected _mouseButton: number
  /** item number */
  protected _item: number

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: ViewerProps) {
    super(props)
    this._container = null
    this._viewerId = -1

    const state = Session.getState()
    if (this.props.id in state.user.viewerConfigs) {
      this._viewerConfig = state.user.viewerConfigs[this.props.id]
    }

    this._keyDownHandler = this.onKeyDown.bind(this)
    this._keyUpHandler = this.onKeyUp.bind(this)
    this._wheelHandler = this.onWheel.bind(this)

    this._keyDownMap = {}
    this._mX = 0
    this._mY = 0
    this._mouseDown = false
    this._mouseButton = -1
    this._item = -1
  }

  /**
   * Run when component mounts
   */
  public componentDidMount () {
    super.componentDidMount()
    document.addEventListener('keydown', this._keyDownHandler)
    document.addEventListener('keyup', this._keyUpHandler)
  }

  /**
   * Run when component unmounts
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    document.removeEventListener('keydown', this._keyDownHandler)
    document.removeEventListener('keyup', this._keyUpHandler)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    this._viewerId = this.props.id
    this._viewerConfig = this.state.user.viewerConfigs[this._viewerId]
    this._item = this.state.user.select.item

    return (
      <div
        ref={(element) => {
          if (element && this._container !== element) {
            if (this._container) {
              this._container.removeEventListener('wheel', this._wheelHandler)
            }
            this._container = element
            this._container.addEventListener('wheel', this._wheelHandler)
            this.forceUpdate()
          }
        }}
        className={this.props.classes.viewer_container}
        onMouseDown={ (e) => this.onMouseDown(e) }
        onMouseUp={ (e) => this.onMouseUp(e) }
        onMouseMove={ (e) => this.onMouseMove(e) }
        onMouseEnter={ (e) => this.onMouseEnter(e) }
        onMouseLeave={ (e) => this.onMouseLeave(e) }
        onDoubleClick={ (e) => this.onDoubleClick(e) }
      >
        {this.getDrawableComponents()}
      </div>
    )
  }

  /** Normalize coordinates to container */
  protected normalizeCoordinates (x: number, y: number): [number, number] {
    if (this._container) {
      const rect = this._container.getBoundingClientRect()
      return [x - rect.left, y - rect.top]
    }
    return [x, y]
  }

  /**
   * Whether a specific key is pressed down
   * @param {string} key - the key to check
   * @return {boolean}
   */
  protected isKeyDown (key: string): boolean {
    return this._keyDownMap[key]
  }

  /** Get child components for rendering */
  protected abstract getDrawableComponents (): React.ReactElement[]

  /**
   * Handle mouse down
   * @param e
   */
  protected onMouseDown (e: React.MouseEvent): void {
    if (!this._container) {
      return
    }
    this._mouseDown = true
    this._mouseButton = e.button
    const normalized = this.normalizeCoordinates(e.clientX, e.clientY)
    this._mX = normalized[0]
    this._mY = normalized[1]
  }

  /**
   * Handle mouse up
   * @param e
   */
  protected onMouseUp (_e: React.MouseEvent): void {
    this._mouseDown = false
  }

  /**
   * Handle mouse move
   * @param e
   */
  protected onMouseMove (e: React.MouseEvent): void {
    const normalized = this.normalizeCoordinates(e.clientX, e.clientY)
    this._mX = normalized[0]
    this._mY = normalized[1]
  }

  /**
   * Handle double click
   * @param e
   */
  protected abstract onDoubleClick (e: React.MouseEvent): void

  /**
   * Handle mouse leave
   * @param e
   */
  protected onMouseEnter (_e: React.MouseEvent) {
    Session.activeViewerId = this.props.id
  }

  /**
   * Handle mouse leave
   * @param e
   */
  protected abstract onMouseLeave (_e: React.MouseEvent): void

  /**
   * Handle mouse wheel
   * @param e
   */
  protected abstract onWheel (e: WheelEvent): void

  /**
   * Handle key down
   * @param e
   */
  protected onKeyUp (e: KeyboardEvent): void {
    delete this._keyDownMap[e.key]
  }

  /**
   * Handle key down
   * @param e
   */
  protected onKeyDown (e: KeyboardEvent): void {
    this._keyDownMap[e.key] = true
  }
}
