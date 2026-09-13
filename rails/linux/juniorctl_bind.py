"""Bind T6/T9/T10 skill-pin helpers onto juniorctl."""
from __future__ import annotations


def install(ns: dict) -> None:
    def _path() -> None:
        ns["_path"]()

    def skill_pin_verify(root=None, skills_dir="skills"):
        _path()
        from rails.linux.ctl_skillpin import skill_pin_verify as impl
        return impl(root, skills_dir)

    def skill_pin_verify_one(rel=None, root=None):
        _path()
        from rails.linux.ctl_skillpin import skill_pin_verify_one as impl
        return impl(rel, root)

    def skill_pin_list(root=None, skills_dir="skills"):
        _path()
        from rails.linux.ctl_skillpin import skill_pin_list as impl
        return impl(root, skills_dir)

    def main(argv):
        _path()
        from rails.linux.ctl_cli import run
        return run(argv, ns)

    ns["skill_pin_verify"] = skill_pin_verify
    ns["skill_pin_verify_one"] = skill_pin_verify_one
    ns["skill_pin_list"] = skill_pin_list
    ns["main"] = main
