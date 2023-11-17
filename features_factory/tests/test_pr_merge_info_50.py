import sys

sys.path.append("..")
from pr_merge_info_50 import (
    check_merge_from_comment,
    get_repo_commit_mention_pull,
    has_sha1_in_sentence,
    is_sha1,
)

if __name__ == "__main__":
    # 下面是测试代码
    # print(check_merge_from_comment('xxx', "fluttergithubbot"))
    # print(check_merge_from_comment("@annthurium are you sure you committed package.json?  Isn't showing up for me in the diff.", "MEMBER"))

    # print(check_merge_from_comment('''\nThanks for your pull request. It looks like this may be your first contribution to a Google open source project (if not, look below for help). Before we can look at your pull request, you'll need to sign a Contributor License Agreement (CLA).\n\n:memo: **Please visit <https://cla.developers.google.com/> to sign.**\n\nOnce you've signed (or fixed any issues), please reply here with `@googlebot I signed it!` and we'll verify it.\n\n----\n\n#### What to do if you already signed the CLA\n\n##### Individual signers\n\n*   It's possible we don't have your GitHub username or you're using a different email address on your commit. Check [your existing CLA data](https://cla.developers.google.com/clas) and verify that your [email is set on your git commits](https://help.github.com/articles/setting-your-email-in-git/).\n\n##### Corporate signers\n\n*   Your company has a Point of Contact who decides which employees are authorized to participate. Ask your POC to be added to the group of authorized contributors. If you don't know who your Point of Contact is, direct the Google project maintainer to [go/cla#troubleshoot](http://go/cla#troubleshoot) ([Public version](https://opensource.google/docs/cla/#troubleshoot)).\n*   The email used to register you as an authorized contributor must be the email used for the Git commit. Check [your existing CLA data](https://cla.developers.google.com/clas) and verify that your [email is set on your git commits](https://help.github.com/articles/setting-your-email-in-git/).\n*   The email used to register you as an authorized contributor must also be [attached to your GitHub account](https://github.com/settings/emails).\n\t\t\n\n\u2139\ufe0f **Googlers: [Go here](https://goto.google.com/prinfo/https%3A%2F%2Fgithub.com%2Fangular%2Fangular%2Fpull%2F43385) for more info**.\n\n<!-- need_sender_cla -->''', "google-cla[bot]"))

    # print(check_merge_from_comment("Superseded by #18300", "CONTRIBUTOR"))

    # print(check_merge_from_comment("Landed in 7e7062cdca87e5ff54945fc1786cba25d0996995.Thanks for the contribution! 🎉"))

    # a = check_merge_from_comment('This PR was merged into the repository by commit 9c03b6371a96c63316623b8820960b4b5d494bb4.')
    # print('wola~', a)

    msg = "lgtm"
    print("lgtm", check_merge_from_comment(msg))
    """
    print(is_sha1('1b47866a1d94224678ce05e7c7355ff17ad9cce6'))
    print(is_sha1('338ab0dfa0c9'))
    print(int('338ab0dfa0c9', 16))

    msg = "Initialize NativeDetector at build time\n\nCloses gh-28244 gh-3333"
    msg = "Change link from 5.3.x to main\n\nSee gh-28228"
    print(get_repo_commit_mention_pull(msg))
    """
    msg = """
        docs(bazel): fix outdated redirect URL for `/guide/bazel` (#43376)

        The file we are redirecting `/guide/bazel` to was moved from
        `bazel/src/schematics/README.md` to `bazel/docs/BAZEL_SCHEMATICS.md` in
        commit 71b8c9ab29014f7e710e03ebda185c0a7c0c2620.

        Update the Firebase configuration to use the new path in the redirect
        URL.

        PR Close #43376
    """
    print(has_sha1_in_sentence(msg))
    msg = """
        build: initialize variable before use\n\nfound with make --warn-undefined-variables\n\nPR-URL: https://github.com/iojs/io.js/pull/320\nReviewed-By: Rod Vagg <rod@vagg.org>
    """
    print(get_repo_commit_mention_pull(msg))

    msg = """
    Drag and drop to install apk files from computer\n\n<https://github.com/Genymobile/scrcpy/pull/133>
    """
    print(get_repo_commit_mention_pull(msg))
