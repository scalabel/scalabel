import { withStyles } from '@material-ui/core/styles'
import _ from 'lodash'
import React from 'react'
import { label2dViewStyle } from '../styles/label'

interface ClassType {
  /** crosshair class */
  hair: string
}

/**
 * Interface used for props.
 */
interface Props {
  /** classes */
  classes: ClassType
  /** container */
  display: HTMLDivElement | null
}

interface State {
  /** x */
  x: number
  /** x */
  y: number
  /** x */
  displayX: number
  /** x */
  displayY: number
  /** x */
  displayW: number
  /** x */
  displayH: number
}

/**
 * Crosshair for 2D annotation
 */
export class Crosshair2D extends React.Component<Props, State> {
  /** horizontal crosshair */
  public h: React.ReactElement | null
  /** vertical crosshair */
  public v: React.ReactElement | null

  constructor (props: Readonly<Props>) {
    super(props)
    this.h = null
    this.v = null
    this.state = {
      x: -1,
      y: -1,
      displayX: -1,
      displayY: -1,
      displayW: -1,
      displayH: -1
    }
  }

  /**
   * ToolBar render function
   * @return component
   */
  public render () {
    const { classes } = this.props
    let valid = false
    valid = this.state.x >= this.state.displayX &&
    this.state.x < this.state.displayX + this.state.displayW &&
    this.state.y >= this.state.displayY &&
    this.state.y < this.state.displayY + this.state.displayH
    if (valid) {
      this.h = <div id='crosshair-h'
                  className={classes.hair}
                  style={{
                    top: this.state.y,
                    left: this.state.displayX,
                    width: this.state.displayW
                  }}
      />
      this.v = <div id='crosshair-v'
                  className={classes.hair}
                  style={{
                    left: this.state.x,
                    top: this.state.displayY,
                    height: this.state.displayH
                  }}
      />
    }

    return (
      <div id='crosshair'
        onMouseMove=
        {
        (e: React.MouseEvent<HTMLElement>) => { this.onMouseMove(e) }
        }
        style={{
          height: '100%',
          width: '100%',
          position: 'absolute'
        }}
      >
        {this.h}
        {this.v}
      </div>
    )
  }

  /**
   * update crosshair
   * @param {number} x
   * @param {number} y
   * @param {number} displayX
   * @param {number} displayY
   * @param {number} displayW
   * @param {number} displayH
   */
  public updateCrosshair (x: number, y: number,
                          displayX: number, displayY: number,
                          displayW: number, displayH: number) {
    this.setState({
      x, y, displayX, displayY, displayW, displayH
    })
  }

  /**
   * update crosshair when mouse moves
   */
  public onMouseMove (e: React.MouseEvent<HTMLElement>) {
    if (this.props.display && this != null) {
      const rect = this.props.display.getBoundingClientRect()
      this.updateCrosshair(e.clientX, e.clientY, rect.left, rect.top,
        rect.width, rect.height)
    }
  }
}

export const Crosshair = withStyles(label2dViewStyle)(Crosshair2D)
