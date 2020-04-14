import { Theme } from '@material-ui/core/styles'
import createStyles from '@material-ui/core/styles/createStyles'

export const accessStyle = (theme: Theme) => createStyles({
  body: {
    width: '100%',
    height: '100%',
    position: 'fixed'
  },
  subLayout: {
    'display': 'flex',
    'flexDirection': 'row',
    'width': '100%',
    'height': '100%',
    'flex': '1 1 auto',
    'minHeight': 'e("calc(100vh - 5.6rem)")',
    '& section': {
      width: '100%'
    },
    '& nav': {
      minWidth: '250px'
    }
  },
  mainContent: {
    paddingLeft: '3rem',
    paddingTop: '1rem'
  },
  grow: {
    marginRight: '2rem'
  },
  toolbar: {
    display: 'flex',
    flexDirection: 'row',
    float: 'right'
  },
  search: {
    'position': 'relative',
    'borderRadius': theme.shape.borderRadius,
    'backgroundColor': 'fade(theme.palette.common.white, 0.15)',
    '&:hover': {
      backgroundColor: 'fade(theme.palette.common.white, 0.25)'
    },
    'marginRight': theme.spacing(2),
    'marginLeft': 0,
    'width': '100%',
    [theme.breakpoints.up('sm')]: {
      marginLeft: theme.spacing(3),
      width: 'auto'
    }
  },
  searchIcon: {
    padding: theme.spacing(0, 2),
    height: '100%',
    position: 'absolute',
    pointerEvents: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  inputRoot: {
    color: 'inherit',
    border: '1px solid',
    borderRadius: '5px'
  },
  inputInput: {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)}px)`,
    transition: theme.transitions.create('width'),
    width: '100%',
    [theme.breakpoints.up('md')]: {
      width: '20ch'
    }
  },
  table: {
    minWidth: 700
  }
})

export const tableCellStyle = (theme: Theme) => createStyles({
  head: {
    backgroundColor: theme.palette.common.black,
    color: theme.palette.common.white
  },
  body: {
    fontSize: 14
  }
})

export const tableRowStyle = (theme: Theme) => createStyles({
  root: {
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.background.default
    }
  }
})
