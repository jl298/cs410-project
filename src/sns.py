import json
import os

BOX_WIDTH = 120


class SNSCommandLine:
    def __init__(self):
        self.commands = {}
        self.is_running = True
        self.posts_file = "../dataset/sns-posts-dataset.json"
        self.user_file = "../dataset/sns-user-dataset.json"
        self.recommendation_file = "../dataset/recommendations-dataset.json"
        self.sentiment_file = "../dataset/sentiment-dataset.json"

        self.posts_per_page = 1
        self.current_user_id = self.get_current_user_id()

        self.register_command("help", self.help_command, "Shows all available commands.")
        self.register_command("quit", self.quit_command, "Exits the program.")
        self.register_command("feed", self.feed_command, "Shows all posts with pagination. Usage: feed [page] [search_term]")
        self.register_command("search", self.search_command, "Search posts by text or product ID. Usage: search <term>")
        self.register_command("post", self.post_command, "Creates a new post with product ID.")
        self.register_command("comment", self.comment_command, "Adds a comment to a post.")
        self.register_command("like", self.like_command, "Adds a like reaction to a post.")
        self.register_command("view", self.view_command, "Adds a view reaction to a post.")
        self.register_command("ignore", self.ignore_command, "Adds an ignore reaction to a post.")
        self.register_command("user", self.user_command, "Shows current user's interaction data.")
        self.register_command("click", self.click_command, "Increments the click count of the post.")
        self.register_command("update", self.update_command,"Creates a new post from highest confidence recommendation.")

    def get_string_width(self, s):
        width = 0
        i = 0
        while i < len(s):
            c = s[i]
            if c in '─│┌┐└┘├┤┄╰►╔╗╚╝║═╮╭':
                width += 1
            elif c == '🏷':
                width += 2
                if i + 1 < len(s) and s[i+1] == '️':
                    i += 1
            elif c in ['📌', '👤', '❤', '👆', '💬', '📝', '►', '✅' ,'🖱' ,'👀️', '🚫']:
                width += 2
                if i + 1 < len(s) and s[i+1] == '️':
                    i += 1
            elif ord(c) > 0x1F000:
                width += 2
            else:
                width += 1
            i += 1
        return width

    def pad_line(self, content, width):
        visible_width = self.get_string_width(content)
        padding_needed = width - visible_width
        return content + ' ' * padding_needed

    def draw_box(self, width):
        horizontal = '─' * (width - 2)
        return {
            'top': f'┌{horizontal}┐',
            'bottom': f'└{horizontal}┘',
            'side': '│',
            'separator': f'├{horizontal}┤'
        }

    def draw_box(self, width):
        horizontal = '═' * (width - 2)
        separator = '─' * (width - 2)
        return {
            'top': f'╔{horizontal}╗',
            'bottom': f'╚{horizontal}╝',
            'side': '║',
            'separator': f'╟{separator}╢'
        }

    def create_line(self, content, width):
        content_width = width - 2
        padded_content = self.pad_line(content, content_width)
        return f"║{padded_content}║"

    def center_text(self, text, width):
        content_width = width - 2
        text_width = self.get_string_width(text)
        left_padding = (content_width - text_width) // 2
        right_padding = content_width - text_width - left_padding
        return f"║{' ' * left_padding}{text}{' ' * right_padding}║"

    def register_command(self, command_name, function, description):
        self.commands[command_name] = {
            'function': function,
            'description': description
        }

    def load_posts(self):
        if not os.path.exists(self.posts_file):
            return []
        with open(self.posts_file, 'r') as file:
            return json.load(file)

    def save_posts(self, posts):
        with open(self.posts_file, 'w') as file:
            json.dump(posts, file, indent=2)

    def load_user_data(self):
        if not os.path.exists(self.user_file):
            user_data = {
                "user_id": self.current_user_id,
                "user_name": "Default User",
                "liked_list": [],
                "viewed_list": [],
                "ignored_list": [],
                "clicked_list": []
            }
            self.save_user_data(user_data)
            return user_data

        with open(self.user_file, 'r') as file:
            return json.load(file)

    def save_user_data(self, user_data):
        with open(self.user_file, 'w') as file:
            json.dump(user_data, file, indent=2)

    def get_current_user_id(self):
        try:
            if os.path.exists(self.user_file):
                with open(self.user_file, 'r') as file:
                    user_data = json.load(file)
                    return user_data.get('user_id', 'default_user')
            return 'default_user'
        except Exception as e:
            print(f"Error reading user ID: {e}")
            return 'default_user'

    def find_post_by_id(self, post_id, posts):
        for post in posts:
            if post['post_id'] == post_id:
                return post
        return None

    def paginate_posts(self, posts, page=1):
        total_posts = len(posts)
        total_pages = (total_posts + self.posts_per_page - 1) // self.posts_per_page

        start_idx = (page - 1) * self.posts_per_page
        end_idx = min(start_idx + self.posts_per_page, total_posts)

        return posts[start_idx:end_idx], total_pages, page

    def search_posts(self, posts, search_term):
        search_term = search_term.lower()
        return [
            post for post in posts
            if search_term in post['post_text'].lower()
               or search_term in post['product_id'].lower()
               or search_term in post['user_id'].lower()
               or any(search_term in comment['comment_text'].lower() for comment in post['comments'])
        ]

    def print_navigation_help(self, current_page, total_pages, search_term=None):
        box = self.draw_box(BOX_WIDTH)
        lines = []
        lines.append(box['top'])
        lines.append(self.center_text("📱 Navigation Commands 📱", BOX_WIDTH))
        lines.append(self.create_line(f"Current Page: {current_page} / {total_pages}", BOX_WIDTH))
        lines.append(self.create_line("", BOX_WIDTH))
        lines.append(self.create_line("Commands:", BOX_WIDTH))
        lines.append(self.create_line("  feed <page>          - Go to specific page", BOX_WIDTH))
        lines.append(self.create_line("  feed next            - Next page", BOX_WIDTH))
        lines.append(self.create_line("  feed prev            - Previous page", BOX_WIDTH))
        lines.append(self.create_line("  search <term>        - Search posts", BOX_WIDTH))
        if search_term:
            lines.append(self.create_line(f"Current search: '{search_term}'", BOX_WIDTH))
        lines.append(box['bottom'])
        print('\n'.join(lines))

    def format_post(self, post, width=BOX_WIDTH):
        box = self.draw_box(width)
        lines = []

        lines.append(box['top'])
        lines.append(self.create_line("", width))
        title = f"📌 Post #{post['post_id']}"
        lines.append(self.center_text(title, width))
        lines.append(self.create_line("", width))
        lines.append(box['separator'])

        lines.append(self.create_line(f"👤 User: {post['user_id']}", width))
        lines.append(box['separator'])

        lines.append(self.create_line(f"🏷️ Product: {post['product_id']}", width))
        lines.append(self.create_line(f"📝 {post['post_text']}", width))
        lines.append(box['separator'])

        lines.append(self.create_line(f"❤️ {post['like_count']} Likes    👀️ {post['view_count']} Views    🚫 {post['ignore_count']} Ignores    👆 {post['click_count']} Clicks", width))

        if post['comments']:
            lines.append(box['separator'])
            lines.append(self.center_text("💬 Comments", width))

            for comment in post['comments']:
                lines.append(self.create_line("┄" * (width - 4), width))

                user_line = f"👤 {comment['user_id']}"
                lines.append(self.create_line(user_line, width))

                wrapped_text = self.wrap_text(comment['comment_text'], width - 6)
                for i, line in enumerate(wrapped_text):
                    prefix = "╰─►" if i == 0 else "   "
                    lines.append(self.create_line(f"{prefix} {line}", width))

        lines.append(box['bottom'])
        return '\n'.join(lines)

    def wrap_text(self, text, max_width):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = self.get_string_width(word)
            if current_width + word_width + (1 if current_line else 0) <= max_width:
                current_line.append(word)
                current_width += word_width + (1 if current_line else 0)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def help_command(self, args):
        box = self.draw_box(BOX_WIDTH)
        lines = []

        lines.append(box['top'])
        lines.append(self.create_line("", BOX_WIDTH))
        title = "Available Commands"
        lines.append(self.center_text(title, BOX_WIDTH))
        lines.append(self.create_line("", BOX_WIDTH))
        lines.append(box['separator'])

        max_cmd_length = max(len(cmd) for cmd in self.commands.keys())

        formatted_commands = []
        for cmd, info in sorted(self.commands.items()):
            padded_cmd = f"{cmd:<{max_cmd_length}}"
            formatted_commands.append(f"  {padded_cmd}  │  {info['description']}")

        for cmd_line in formatted_commands:
            lines.append(self.create_line(cmd_line, BOX_WIDTH))

        lines.append(box['bottom'])
        print('\n'.join(lines))

    def quit_command(self, args):
        print("👋 Good bye.")
        self.is_running = False

    def update_command(self, args):
        try:
            if not os.path.exists(self.recommendation_file):
                print(f"Recommendation dataset not found: {self.recommendation_file}")
                return

            with open(self.recommendation_file, 'r') as file:
                recommendations = json.load(file)

            if not recommendations:
                print("No recommendations found in dataset.")
                return

            highest_conf_rec = max(recommendations, key=lambda x: float(x['confidence']))

            posts = self.load_posts()
            max_post_id = 0
            if posts:
                max_post_id = max(int(post['post_id']) for post in posts)

            new_post = {
                'post_id': str(max_post_id + 1),
                'user_id': self.current_user_id,
                'product_id': highest_conf_rec['recommended_product'],
                'post_text': f"Recommended product based on your interest in {highest_conf_rec['source_product']}! Confidence: {highest_conf_rec['confidence']}",
                'like_count': 0,
                'view_count': 0,
                'ignore_count': 0,
                'click_count': 0,
                'comments': [],
                'tag': 'online'
            }

            posts.append(new_post)

            for post in posts:
                if 'tag' in post and post['tag'] == 'online':
                    post['tag'] = 'offline'

                for comment in post['comments']:
                    if 'tag' in comment and comment['tag'] == 'online':
                        comment['tag'] = 'offline'

            self.save_posts(posts)

            if os.path.exists(self.sentiment_file):
                try:
                    with open(self.sentiment_file, 'r') as file:
                        sentiment_data = json.load(file)

                    for item in sentiment_data:
                        if 'tag' in item and item['tag'] == 'online':
                            item['tag'] = 'offline'

                    with open(self.sentiment_file, 'w') as file:
                        json.dump(sentiment_data, file, indent=2)
                except Exception as e:
                    print(f"Error updating sentiment dataset: {str(e)}")

            box = self.draw_box(BOX_WIDTH)
            print(box['top'])
            print(self.center_text("✅ Post Created from Recommendation", BOX_WIDTH))
            print(self.create_line(f"Post ID: {new_post['post_id']}", BOX_WIDTH))
            print(self.create_line(f"Product ID: {new_post['product_id']}", BOX_WIDTH))
            print(self.create_line(f"Based on: {highest_conf_rec['source_product']}", BOX_WIDTH))
            print(self.create_line(f"Confidence: {highest_conf_rec['confidence']}", BOX_WIDTH))
            print(self.create_line("All online tags updated to offline", BOX_WIDTH))
            print(box['bottom'])

        except Exception as e:
            print(f"Error updating from recommendations: {str(e)}")

    def post_command(self, args):
        product_id = input("Enter product ID: ")
        post_text = input("Enter post text: ")

        posts = self.load_posts()

        max_post_id = 0
        if posts:
            max_post_id = max(int(post['post_id']) for post in posts)

        new_post = {
            'post_id': str(max_post_id + 1),
            'user_id': self.current_user_id,
            'product_id': product_id,
            'post_text': post_text,
            'like_count': 0,
            'view_count': 0,
            'ignore_count': 0,
            'click_count': 0,
            'comments': [],
            'tag': 'online'
        }

        posts.append(new_post)
        self.save_posts(posts)

        box = self.draw_box(BOX_WIDTH)
        print(box['top'])
        print(self.center_text("✅ Post Created Successfully", BOX_WIDTH))
        print(self.create_line(f"Post ID: {new_post['post_id']}", BOX_WIDTH))
        print(self.create_line(f"Product ID: {product_id}", BOX_WIDTH))
        print(box['bottom'])

    def search_command(self, args):
        if not args:
            print("Please provide a search term. Usage: search <term>")
            return

        search_term = ' '.join(args)
        posts = self.load_posts()
        filtered_posts = self.search_posts(posts, search_term)

        if not filtered_posts:
            print(f"No posts found matching '{search_term}'")
            return

        self.display_feed(filtered_posts, 1, search_term)

    def feed_command(self, args):
        posts = self.load_posts()
        if not posts:
            print("No posts found.")
            return

        current_page = 1
        search_term = None

        if args:
            if args[0].isdigit():
                current_page = int(args[0])
            elif args[0] in ['next', 'prev']:
                last_page = self.load_last_page()
                if args[0] == 'next':
                    current_page = min(last_page + 1, (len(posts) + self.posts_per_page - 1) // self.posts_per_page)
                else:
                    current_page = max(1, last_page - 1)
            else:
                search_term = ' '.join(args)
                posts = self.search_posts(posts, search_term)

        if search_term and not posts:
            print(f"No posts found matching '{search_term}'")
            return

        self.display_feed(posts, current_page, search_term)

    def display_feed(self, posts, page, search_term=None):
        paginated_posts, total_pages, current_page = self.paginate_posts(posts, page)

        if not paginated_posts:
            print(f"No posts found on page {page}")
            return

        self.save_last_page(current_page)

        box = self.draw_box(BOX_WIDTH)
        print("\n" + box['top'])
        print(self.center_text("📱 Social Feed 📱", BOX_WIDTH))
        if search_term:
            print(self.center_text(f"Search: '{search_term}'", BOX_WIDTH))
        print(self.center_text(f"Page {current_page} of {total_pages}", BOX_WIDTH))
        print(box['bottom'])

        for post in paginated_posts:
            print("\n" + self.format_post(post))

        self.print_navigation_help(current_page, total_pages, search_term)

    def load_last_page(self):
        try:
            with open('.last_page', 'r') as f:
                return int(f.read().strip())
        except:
            return 1

    def save_last_page(self, page):
        with open('.last_page', 'w') as f:
            f.write(str(page))

    def comment_command(self, args):
        post_id = input("Enter post ID: ")
        comment_text = input("Enter comment text: ")
        reaction = input("Enter reaction (liked/viewed/ignored): ").lower()

        if reaction not in ['liked', 'viewed', 'ignored']:
            print("Invalid reaction. Must be 'liked', 'viewed', or 'ignored'")
            return

        posts = self.load_posts()
        post = self.find_post_by_id(post_id, posts)

        if not post:
            print(f"Post with ID {post_id} not found.")
            return

        max_comment_id = 0
        if post['comments']:
            max_comment_id = max(int(comment['comment_id']) for comment in post['comments'])

        new_comment = {
            'comment_id': str(max_comment_id + 1),
            'user_id': self.current_user_id,
            'comment_text': comment_text,
            'reaction': reaction,
            'tag': 'online'
        }

        post['comments'].append(new_comment)
        self.update_post_counts(post, reaction)
        self.update_user_lists(post['product_id'], reaction)

        user_data = self.load_user_data()
        user_data['post_id_of_last_comment'] = post_id
        user_data['last_comment_id'] = new_comment['comment_id']
        self.save_user_data(user_data)

        self.save_posts(posts)

        box = self.draw_box(BOX_WIDTH)
        print(box['top'])
        print(self.center_text("✅ Comment Added Successfully", BOX_WIDTH))
        print(box['bottom'])

    def update_post_counts(self, post, reaction):
        if reaction == 'liked':
            post['like_count'] += 1
        elif reaction == 'viewed':
            post['view_count'] += 1
        elif reaction == 'ignored':
            post['ignore_count'] += 1

    def update_user_lists(self, product_id, reaction):
        user_data = self.load_user_data()

        if reaction == 'liked' and product_id not in user_data['liked_list']:
            user_data['liked_list'].append(product_id)
        elif reaction == 'viewed' and product_id not in user_data['viewed_list']:
            user_data['viewed_list'].append(product_id)
        elif reaction == 'ignored' and product_id not in user_data['ignored_list']:
            user_data['ignored_list'].append(product_id)

        self.save_user_data(user_data)

    def click_command(self, args):
        post_id = input("Enter post ID: ")

        posts = self.load_posts()
        post = self.find_post_by_id(post_id, posts)

        if not post:
            print(f"Post with ID {post_id} not found.")
            return

        post['click_count'] += 1
        self.save_posts(posts)

        user_data = self.load_user_data()
        product_id = post['product_id']

        if product_id not in user_data['clicked_list']:
            user_data['clicked_list'].append(product_id)
            self.save_user_data(user_data)

        box = self.draw_box(BOX_WIDTH)
        lines = []
        lines.append(box['top'])
        lines.append(self.center_text("✅ Click Registered", BOX_WIDTH))
        lines.append(
            self.create_line(f"New click count: {post['click_count']} for product: {post['product_id']}", BOX_WIDTH))
        lines.append(self.create_line(f"Updated user's clicked products", BOX_WIDTH))
        lines.append(box['bottom'])
        print('\n'.join(lines))

    def like_command(self, args):
        self.reaction_command(args, 'liked')

    def view_command(self, args):
        self.reaction_command(args, 'viewed')

    def ignore_command(self, args):
        self.reaction_command(args, 'ignored')

    def reaction_command(self, args, reaction_type):
        post_id = input("Enter post ID: ")

        posts = self.load_posts()
        post = self.find_post_by_id(post_id, posts)

        if not post:
            print(f"Post with ID {post_id} not found.")
            return

        self.update_post_counts(post, reaction_type)
        self.update_user_lists(post['product_id'], reaction_type)
        self.save_posts(posts)

        box = self.draw_box(BOX_WIDTH)
        lines = []
        lines.append(box['top'])
        lines.append(self.center_text(f"✅ {reaction_type.capitalize()} Added", BOX_WIDTH))

        if reaction_type == 'liked':
            count = post['like_count']
        elif reaction_type == 'viewed':
            count = post['view_count']
        else:
            count = post['ignore_count']

        lines.append(
            self.create_line(f"New {reaction_type} count: {count} for product: {post['product_id']}", BOX_WIDTH))
        lines.append(self.create_line(f"Updated user's {reaction_type} products", BOX_WIDTH))
        lines.append(box['bottom'])
        print('\n'.join(lines))

    def format_product_list(self, products, width):
        if not products:
            return ["None"]

        lines = []
        current_line = []
        current_width = 0

        for product in products:
            item_width = len(product) + 2

            if not current_line or current_width + item_width <= width - 4:
                current_line.append(product)
                current_width += item_width
            else:
                lines.append(", ".join(current_line))
                current_line = [product]
                current_width = item_width

        if current_line:
            lines.append(", ".join(current_line))

        return lines

    def user_command(self, args):
        user_data = self.load_user_data()
        box = self.draw_box(BOX_WIDTH)
        lines = []

        lines.append(box['top'])
        lines.append(self.create_line("", BOX_WIDTH))
        lines.append(self.center_text("👤 User Profile", BOX_WIDTH))
        lines.append(self.create_line("", BOX_WIDTH))

        lines.append(box['separator'])
        lines.append(self.create_line(f"User ID: {user_data['user_id']}", BOX_WIDTH))
        lines.append(self.create_line(f"User Name: {user_data['user_name']}", BOX_WIDTH))

        lines.append(box['separator'])
        lines.append(self.center_text("❤️ Liked Products", BOX_WIDTH))
        for line in self.format_product_list(user_data['liked_list'], BOX_WIDTH - 10):
            lines.append(self.create_line(f"  {line}", BOX_WIDTH))
        lines.append(self.create_line(f"Total Likes: {len(user_data['liked_list'])}", BOX_WIDTH))

        lines.append(box['separator'])
        lines.append(self.center_text("👀 Viewed Products", BOX_WIDTH))
        for line in self.format_product_list(user_data['viewed_list'], BOX_WIDTH - 10):
            lines.append(self.create_line(f"  {line}", BOX_WIDTH))
        lines.append(self.create_line(f"Total Views: {len(user_data['viewed_list'])}", BOX_WIDTH))

        lines.append(box['separator'])
        lines.append(self.center_text("🚫 Ignored Products", BOX_WIDTH))
        for line in self.format_product_list(user_data['ignored_list'], BOX_WIDTH - 10):
            lines.append(self.create_line(f"  {line}", BOX_WIDTH))
        lines.append(self.create_line(f"Total Ignores: {len(user_data['ignored_list'])}", BOX_WIDTH))

        lines.append(box['separator'])
        lines.append(self.center_text("👆 Clicked Products", BOX_WIDTH))
        for line in self.format_product_list(user_data['clicked_list'], BOX_WIDTH - 10):
            lines.append(self.create_line(f"  {line}", BOX_WIDTH))
        lines.append(self.create_line(f"Total Clicks: {len(user_data['clicked_list'])}", BOX_WIDTH))

        lines.append(box['bottom'])
        print('\n'.join(lines))

    def parse_input(self, user_input):
        parts = user_input.strip().split()
        if not parts:
            return None, []
        return parts[0].lower(), parts[1:]

    def run(self):
        box = self.draw_box(BOX_WIDTH)

        print(box['top'])
        print(self.center_text("🚀 Starting SNS UI", BOX_WIDTH))
        print(self.center_text(f"Current User: {self.current_user_id}", BOX_WIDTH))
        print(self.center_text("Type 'help' for commands", BOX_WIDTH))
        print(box['bottom'])

        while self.is_running:
            try:
                user_input = input("\n> ")
                command, args = self.parse_input(user_input)

                if command in self.commands:
                    self.commands[command]['function'](args)
                elif command is not None:
                    print(f"Unknown command: {command}")
                    print("Type 'help' to see available commands.")

            except KeyboardInterrupt:
                print("\nType 'quit' to exit the program.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    cli = SNSCommandLine()
    cli.run()