export interface ShapeStyle {
  /** radius of handle */
  handleRadius: number,
  /** when the handle is in highlight */
  handleHighlightRadius: number,
  /** line width */
  lineWidth: number
}

export interface ColorStyle {
  /** alpha of handle */
  handleAlpha: number,
  /** alpha of the line */
  lineAlpha: number
}

export const DEFAULT_VIEW_SHAPE_STYLE = {
  handleRadius: 8,
  handleHighlightRadius: 12,
  lineWidth: 4
}

export const DEFAULT_CONTROL_SHAPE_STYLE = {
  handleRadius: 12,
  handleHighlightRadius: 12,
  lineWidth: 10
}
