import mysql.connector


class User:
    def __init__(self, username, password, firstname, lastname, hotel_name):
        self.username = username
        self.password = password
        self.firstname = firstname
        self.lastname = lastname
        self.hotel_name = hotel_name


class UserAuthenticator:

    def validate_user(self, username, password):
        result = self.get_user(username)

        if result[3] == password:
            return True
        else:
            return False

    def get_user(self, username):
        conn = mysql.connector.connect(host='localhost',
                                   database='comparator_dash',
                                   user='root',
                                   password='')

        cursor = conn.cursor()
        query = "SELECT * FROM user WHERE username = '"+username+"'"
        cursor.execute(query)
        result = cursor.fetchone()
        conn.commit()

        return result

    def set_user(self, user):
        conn = mysql.connector.connect(host='localhost',
                                   database='comparator_dash',
                                   user='root',
                                   password='')

        cursor = conn.cursor()
        query = "INSERT INTO user (firstname, lastname, username, password, hotel_name) VALUES (%s, %s, %s, %s, %s)"
        val = (user.firstname, user.lastname, user.username, user.password, user.hotel_name)

        try:
            cursor.execute(query, val)
            conn.commit()
            status = 'User successfully registered'
        except:
            status = 'Error when registering user'
        finally:
            conn.close()

        return status

    def change_password(self, username, password):
        conn = mysql.connector.connect(host='localhost',
                                       database='comparator_dash',
                                       user='root',
                                       password='')

        cursor = conn.cursor()
        query = "UPDATE user SET password = %s WHERE username = %s"
        val = (password, username)

        try:
            cursor.execute(query, val)
            conn.commit()
            status = 'Password changed successfully'
        except:
            status = 'Error when changing password'
        finally:
            conn.close()

        return status
