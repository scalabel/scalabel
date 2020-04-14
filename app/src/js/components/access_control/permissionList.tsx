import Button from '@material-ui/core/Button'
import InputBase from '@material-ui/core/InputBase'
import Paper from '@material-ui/core/Paper'
import { withStyles } from '@material-ui/core/styles'
import Table from '@material-ui/core/Table'
import TableBody from '@material-ui/core/TableBody'
import TableCell from '@material-ui/core/TableCell'
import TableContainer from '@material-ui/core/TableContainer'
import TableHead from '@material-ui/core/TableHead'
import TableRow from '@material-ui/core/TableRow'
import SearchIcon from '@material-ui/icons/Search'
import { RouteComponentProps } from '@reach/router'
import React from 'react'
import { accessStyle, tableCellStyle, tableRowStyle } from '../../styles/access'

interface ClassType {
  /** grow style */
  grow: string,
  /** search style */
  search: string,
  /** search icon style */
  searchIcon: string,
  /** input root style */
  inputRoot: string,
  /** input input style */
  inputInput: string,
  /** table style */
  table: string,
  /** toolbar style */
  toolbar: string
}

interface Props extends RouteComponentProps {
  /**
   * Render class
   */
  classes: ClassType
}

interface Permission {
  /**
   * Permission name
   */
  name: string,
  /**
   * Permission description
   */
  description: string
}

interface State {
  /**
   * Permission list
   */
  data: Permission[]
}

/**
 * Permission list
 */
class PermissionList extends React.Component<Props, State> {
  constructor (props: Props) {
    super(props)
    this.state = {
      data: []
    }
  }

  /**
   * public render method
   */
  public render () {
    const { classes } = this.props
    return (
      <div>
        <div className={classes.toolbar}>
          <div className={classes.search}>
            <div className={classes.searchIcon}>
              <SearchIcon />
            </div>
            <InputBase
              placeholder='Search…'
              classes={{
                root: classes.inputRoot,
                input: classes.inputInput
              }}
              inputProps={{ 'aria-label': 'search' }}
            />
          </div>
          <div className={classes.grow} >
            <Button variant='contained'>
              Add Permission
            </Button>
          </div>
        </div>
        {this.state.data.length > 0 && (
        <TableContainer component={Paper}>
          <Table className={classes.table} aria-label='customized table'>
            <TableHead>
              <TableRow>
                <StyledTableCell>Name</StyledTableCell>
                <StyledTableCell align='right'>Description</StyledTableCell>
                <StyledTableCell align='right'>Actions</StyledTableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {
                this.state.data.map((row) => (
                  <StyledTableRow key={row.name}>
                    <StyledTableCell component='th' scope='row'>
                      {row.name}
                    </StyledTableCell>
                    <StyledTableCell align='right'>
                      {row.description}
                    </StyledTableCell>
                    <StyledTableCell align='right'>
                      <Button>

                      </Button>
                      <Button>

                      </Button>
                    </StyledTableCell>
                  </StyledTableRow>
                ))
              }
            </TableBody>
          </Table>
        </TableContainer>
        )}
      </div>
    )
  }
}

const StyledTableRow =
  withStyles(tableRowStyle, { withTheme: true })(TableRow)
const StyledTableCell =
  withStyles(tableCellStyle, { withTheme: true })(TableCell)
export default withStyles(accessStyle, { withTheme: true })(PermissionList)
