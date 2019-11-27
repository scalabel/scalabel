import createStyles from '@material-ui/core/styles/createStyles'

export const resizerStyles = () => createStyles({
  resizer: {
    'background': 'rgba(150, 150, 150, 1)',
    'opacity': 0.2,
    'zIndex': 1,
    '-moz-box-sizing': 'border-box',
    '-webkit-box-sizing': 'border-box',
    'box-sizing': 'border-box',
    '-moz-background-clip': 'padding',
    '-webkit-background-clip': 'padding',
    'background-clip': 'padding-box',
    '&:hover': {
      '-webkit-transition': 'all 2s ease',
      'transition': 'all 2s ease'
    },
    '&.horizontal': {
      'height': '11px',
      'margin': '-5px 0',
      'border-top': '5px solid rgba(255, 255, 255, 0)',
      'border-bottom': '5px solid rgba(255, 255, 255, 0)',
      'cursor': 'row-resize',
      'width': '100%'
    },
    '&.vertical': {
      'width': '11px',
      'margin': '0 -5px',
      'border-left': '5px solid rgba(255, 255, 255, 0)',
      'border-right': '5px solid rgba(255, 255, 255, 0)',
      'cursor': 'col-resize',
      '&:hover': {
        'border-top': '5px solid rgba(100, 100, 100, 1)',
        'border-bottom': '5px solid rgba(100, 100, 100, 1)'
      }
    },
    '&.disabled': {
      'cursor': 'not-allowed',
      '&:hover': {
        'border-color': 'transparent'
      }
    }
  }
})
