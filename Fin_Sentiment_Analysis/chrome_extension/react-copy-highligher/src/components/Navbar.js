import {Link} from 'react-router-dom';

const Navbar = () => {
    return ( 
        <nav className="navbar">
            <div className="links">
                <Link to="/">New Jot</Link>
                <Link to="/notepad">Notepad</Link>
            </div>
        </nav>
     );
}
 
export default Navbar;