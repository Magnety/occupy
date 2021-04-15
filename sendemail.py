import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart

smtpserver = 'smtp.qq.com'
username = 'mc-yao@qq.com'
password = 'auxwnwzdrtlfbihb'
sender = 'mc-yao@qq.com'

receiver = ['liuyiyao0916@163.com']
subject = 'test test'
msg = MIMEMultipart('mixed')
msg['Subject'] = subject
msg['From'] = 'mc-yao@qq.com <mc-yao@qq.com>'
msg['To'] = ";".join(receiver)
text = "Hi!\nHow are you?\nHere is the link you wanted:\nhttp://www.baidu.com"
text_plain = MIMEText(text,'plain', 'utf-8')
msg.attach(text_plain)
smtp = smtplib.SMTP()
smtp.connect('smtp.qq.com')
#我们用set_debuglevel(1)就可以打印出和SMTP服务器交互的所有信息。
#smtp.set_debuglevel(1)
smtp.login(username, password)
smtp.sendmail(sender, receiver, msg.as_string())
smtp.quit()

