import React from 'react';
import PropTypes from 'prop-types';
import {AppBar, Toolbar, IconButton, Tooltip}
       from '@material-ui/core';
import {faCheck, faInfo, faQuestion, faSave, faList, faColumns}
       from '@fortawesome/free-solid-svg-icons/index';
import {withStyles} from '@material-ui/core/styles/index';
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';

const styles = (theme) => ({
  appBar: {
    position: 'relative',
    background: '#222222',
    height: '50px',
  },
  grow: {
    flexGrow: 1,
  },
  titleUnit: {
    color: '#bbbbbb',
    margin: theme.spacing.unit * 0.5,
  },
});

type Props = {
  classes: Object,
  theme: Object,
  title: String,
  dashboardLink: String,
  instructionLink: String,
}

/**
 * Title bar
 */
class TitleBar extends React.Component<Props> {
  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  render() {
    const {classes} = this.props;
    const {title} = this.props;
    const {instructionLink} = this.props;
    const {dashboardLink} = this.props;
    return (<AppBar position="static" className={classes.appBar}>
      <Toolbar variant="dense">
        {title}
        <div className={classes.grow} />
        <Tooltip title='Instructions'>
          <IconButton className={classes.titleUnit}
                      href={instructionLink} target="view_window">
            <FontAwesomeIcon icon={faInfo} size='xs'/>
          </IconButton>
        </Tooltip>
        <Tooltip title='Keyboard Usage'>
          <IconButton className={classes.titleUnit}>
            <FontAwesomeIcon icon={faQuestion} size='xs'/>
          </IconButton>
        </Tooltip>
        <Tooltip title='Dashboard'>
          <IconButton className={classes.titleUnit}
                      href={dashboardLink} target="view_window">
            <FontAwesomeIcon icon={faList} size='xs'/>
          </IconButton>
        </Tooltip>
        <Tooltip title='Assistant View'>
          <IconButton className={classes.titleUnit}>
            <FontAwesomeIcon icon={faColumns} size='xs'/>
          </IconButton>
        </Tooltip>
        <Tooltip title='Save'>
          <IconButton className={classes.titleUnit}>
            <FontAwesomeIcon icon={faSave} size='xs'/>
          </IconButton>
        </Tooltip>
        <Tooltip title='Submit'>
          <IconButton className={classes.titleUnit}>
            <FontAwesomeIcon icon={faCheck} size='xs'/>
          </IconButton>
        </Tooltip>
      </Toolbar>
    </AppBar>
    );
  }
}

TitleBar.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
  title: PropTypes.string.isRequired,
  dashboardLink: PropTypes.string.isRequired,
  instructionLink: PropTypes.string.isRequired,
};

export default withStyles(styles, {withTheme: true})(TitleBar);
